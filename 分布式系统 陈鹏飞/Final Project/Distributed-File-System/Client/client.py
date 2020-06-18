import os
import grpc
import FileServer.FileServer_pb2 as FileServer_pb2, FileServer.FileServer_pb2_grpc as FileServer_pb2_grpc
import DirectoryServer.DirectoryServer_pb2 as DS_pb2, DirectoryServer.DirectoryServer_pb2_grpc as DS_pb2_grpc
import LockServer.LockServer_pb2 as LS_pb2, LockServer.LockServer_pb2_grpc as LS_pb2_grpc

from setting import *

class FileClient:
    def __init__(self,id):
        self.id = id
        self.root_path = '../clientData/Client_%s/'%(id)   # 客户端系统的根目录 为了简化文件系统，用户只有根目录不再创建子文件夹
        self.current_path = None # 客户端当前访问服务器的文件夹路径  初始为空
        self.CHUNK_SIZE = 1024 * 1024  # 1MB  传输文件的时候每次传1MB
        self.openfile_list=[]  # 客户端维护一个本地打开的文件队列


    def connect_dirserver(self):
        # 客户端首先连接到目录服务器  然后根据返回的文件服务器列表选择一个文件服务器
        dirchannel = grpc.insecure_channel(dirServer_ip + ":" + dirServer_port)
        self.dirstub = DS_pb2_grpc.DirServerStub(dirchannel)


    def connect_lockserver(self):
        # 客户接着需要连接到锁服务器
        lockchannel = grpc.insecure_channel(lockServer_ip + ":" + lockServer_port)
        self.lockstub = LS_pb2_grpc.LockServerStub(lockchannel)


    def choose_server(self):
        # 连接到文件锁服务器：
        print("Connecting to the file lock server...")
        self.connect_lockserver()

        # 连接目录服务器
        print("Connecting to the directory server...")
        self.connect_dirserver()

        # 从目录服务器中获取所有FS的信息
        while True:
            try:
                response = self.dirstub.getfileserver(DS_pb2.Dir_Empty(empty=0))
                break
            except:
                print("Directory server connection timed out!")
        print("Select one of the following servers:")
        server_dic={}
        title = '{: <15}{: <15}{: <15}{: <15}'.format('ServerID','IP','port','location')
        print(title)
        for server in response.server_list:
            server_dic[server.Server_ID] = {'ip':server.ip,'port':server.port,'location':server.location}
            info = '{: <15}{: <15}{: <15}{: <15}'.format(str(server.Server_ID),server.ip,server.port,server.location)
            print(info)
        while True:
            pick = int(input("Input ServerID to choose: "))
            if pick not in server_dic.keys():
                print("Server does not exist, please select another server!")
                continue

            print("Connecting to file server...")
            try:
                # 记录连接的文件服务器地址
                self.FS_ip = server_dic[pick]['ip']
                self.FS_port = server_dic[pick]['port']
                channel = grpc.insecure_channel(self.FS_ip + ":" + self.FS_port)
                self.stub = FileServer_pb2_grpc.FileServerStub(channel)

                # 获取服务器的根目录地址，同时也可以检测服务器是否在线
                response = self.stub.pwd(FileServer_pb2.Empty(empty=0))
                self.current_path = response.pwd
                break
            except Exception as e:
                print("Server connection timed out, please select another server!")
                continue



    def save_chunks_to_file(self,chunks, target):
        with open(target, 'wb') as f:
            for chunk in chunks:
                f.write(chunk.buffer)


    def get_file_chunks(self,source,target):
        with open(source, 'rb') as f:
            while True:
                piece = f.read(self.CHUNK_SIZE)
                if len(piece) == 0:
                    return
                yield FileServer_pb2.Upload_Request(buffer=piece,target_path = target)  # 同时传文件和路径



    def upload(self, filename):
        # 用户上传文件， 上传完之后文件服务器需要同步到所有的副本上
        source = self.root_path + filename  # 上传文件的原地址
        target = self.current_path + filename  # 上传文件的目标地址 （服务器的地址）

        if self.lockfile(self.current_path, filename) != 0:  # 对文件申请加锁
            print("The file is being edited by someone else")
            return
        if not os.path.exists(source):
            print("files does not exist")
            return

        chunks_generator = self.get_file_chunks(source, target)
        response = self.stub.upload(chunks_generator)
        self.unlockfile(self.current_path, filename)# 解锁文件
        if response.success==0:
            print("File uploaded successfully!")
            return
        else:
            print("File upload failed!")
            return





    def download(self, filename):
        source = self.current_path + filename  # 远程服务器文件路径
        target = self.root_path + filename  # 下载到本地的路径
        source_filelist = self.list() #  获取远程服务器当前目录的文件，判断是否有目标文件

        if not filename in source_filelist: # 如果不存在就退出
            print("Target file does not exist")
            return

        if self.lockfile(self.current_path,filename) != 0: # 对文件申请加锁
            print("The file is being edited by someone else")
            return

        response = self.stub.download(FileServer_pb2.Download_Request(download_path = source))
        if os.path.exists(target):
            # # 如果目标文件已经存在，就在名字后面加个 - copy n  n表示第几个副本
            tmp1 = os.listdir(self.root_path) # 获取当前所有文件名列表
            n=1
            for a in tmp1:  # 如果这个文件名字被当前文件包含，n就+1，  注意：文件名只有后缀才允许包含点
                if filename.split('.')[-2] in a and 'rename' in a:
                    n+=1
            ### 根据副本号n 生成新的目标文件名
            tmp2 = filename.split('.')
            tmp2[-2]+=" - rename%s"%(n)
            target = self.root_path + '.'.join(tmp2)
        self.save_chunks_to_file(response, target)
        print("File downloaded successfully")

        self.unlockfile(self.current_path,filename) # 将文件解锁



    def delete(self,filename):
        ### 用户发消息删除文件服务器上的对应文件名
        target_filelist = self.list()  # 获取远程服务器当前目录的文件，判断是否有目标文件
        if not filename in target_filelist: # 如果不存在就退出
            print("Target file does not exist")
            return
        response = self.stub.delete(FileServer_pb2.Delete_Request(delete_path = self.current_path+filename))
        if response.success == 0:
            print("File deleted successfully")
        else:
            print("File deleted failed")


    def open(self,filename):
        '''
        允许用户打开并编辑一个文件，因为没有像vim一样的编辑器，所以只能模拟一个打开编辑的过程，逻辑如下：
        open相当于用户对一个文件进行download，并对文件加锁。
        但这个锁不会再download后释放，而是一直保存直到用户主动输入close指令。
        这个过程就模拟了用户打开并开始编辑文件的过程，不过打开关闭都需要用户输入指令完成
        客户端会维护一个open的列表保存用户打开的所有进程用户可以查看当前所有被打开的进程
        '''
        target_filelist = self.list()  # 获取远程服务器当前目录的文件，判断是否有目标文件
        if not filename in target_filelist:  # 如果不存在就退出
            print("Target file does not exist")
            return

        source = self.current_path + filename  # 远程服务器文件路径
        target = self.root_path + filename  # 下载到本地的路径
        source_filelist = self.list()  # 获取远程服务器当前目录的文件，判断是否有目标文件
        if not filename in source_filelist:  # 如果不存在就退出
            print("Target file does not exist")
            return

        if self.lockfile(self.current_path, filename) != 0:  # 对文件申请加锁
            print("The file is being edited by someone else")
            return

        response = self.stub.download(FileServer_pb2.Download_Request(download_path=source))

        if os.path.exists(target):
            print("Warning: Local file with the same name is overwritten!")
        self.save_chunks_to_file(response, target)
        print("File open successfully")

        # 打开文件成功后将文件加入到已打开文件队列中去
        self.openfile_list.append((self.current_path,filename))




    def close(self,id):
        '''
        当用户编辑完一个文件后，输入close指令关闭对应文件，对应文件会被upload上传到远程服务器，此时需要上文件锁
        远程服务器接收到上传的文件后，会将上传文件的消息传播到每个副本上。
        传播完成后服务器会返回一个success，此时客户端可以和锁服务器申请取消掉对应文件锁
        '''
        try:
            id = int(id)
            filepath,filename = self.openfile_list[id-1]  # id从1开始
            self.openfile_list.pop(id - 1)  # 将文件从打开列表中取出
            # self.unlockfile(filepath,filename)  # 对加锁文件解锁
            print("File closed successfully.")

            ## 将目标文件更新到远程服务器上
            source = self.root_path + filename  # 上传文件的原地址
            target = filepath + filename  # 上传文件的目标地址 （服务器的地址）

            # if self.lockfile(filepath, filename) != 0:  # 对文件申请加锁
            #     print("The file is being edited by someone else")
            #     return

            if not os.path.exists(source):
                print("files does not exist")
                return
            chunks_generator = self.get_file_chunks(source, target)
            response = self.stub.upload(chunks_generator)

            self.unlockfile(filepath, filename)  # 解锁文件

            if response.success == 0:
                print("File update successfully!")
                return
            else:
                print("File update failed!")
                return

        except Exception as e:
            print(e)
            print("File closed failed.")



    def mkdir(self,dirname):
        '''
        创建文件夹 传dirname然后进行创建
        '''
        target_filelist = self.list()  # 获取远程服务器当前目录的文件，判断是否有目标文件
        if dirname in target_filelist:  # 如果存在就退出
            print("Target directory existed")
            return
        response = self.stub.mkdir(FileServer_pb2.Mkdir_Request(dir_path = self.current_path + dirname))
        if response.success==0:
            print("Folder make successfully")
        else:
            print("Folder make failed")




    def list(self):
        ## 列出当前路径下的所有文件名 类似linux 的ls语句
        response = self.stub.list(FileServer_pb2.List_Request(cur_path = self.current_path))
        return response.list



    def cd(self,dirname):
        ## 访问目标文件夹
        target_filelist = self.list()  # 获取远程服务器当前目录的文件，判断是否有目标目录
        if not dirname in target_filelist:  # 如果存在就退出
            print("Target directory does not exist")
            return
        else:
            self.current_path+=dirname+'/'  # 为了简单就直接进入了，可以用os.path.isdir(path)发消息判断是否是文件夹


    def cdb(self):
        self.current_path=os.path.dirname(self.current_path[:-1])+'/'  #可以返回父亲目录的路径，因为现在结尾是/ ，要把这个/去掉再去父亲目录地址。



    def lockfile(self, filepath,filename):
        # 客户在下载文件或者打开文件的时候会对文件上锁，防止出现写写冲突
        response = self.lockstub.lockfile(LS_pb2.lockfileinfo(file_path=filepath, filename = filename, client_id = self.id))
        return response.success



    def unlockfile(self,filepath,filename):
        response = self.lockstub.unlockfile(LS_pb2.unlockfileinfo(file_path=filepath, filename = filename))
        return response.success


    def listopen(self):
        # 将打开的所有进程列出给用户选择
        title = '{: <15}{: <15}{: <15}'.format('processID', 'filepath', 'filename')
        print(title)
        for index,record in enumerate(self.openfile_list):
            info = '{: <15}{: <15}{: <15}'.format(str(index+1), record[0], record[1])
            print(info)

    def showhelpmesg(self):
        print("-------------------------------------------")
        print("The available commands are as follows：")
        print("ls: list file directories")
        print("lsc: list user local file directories")
        print("upload or u: upload files")
        print("download or d: download files")
        print("pwd: current access server path")
        print("delete: delete files")
        print("mkdir: create folder")
        print("cd: change current path")
        print("cd..: back to previous path")
        print("lso: list open files")
        print("open: open files")
        print("close: close files")
        print("------------------------------------------")




def run_client(id):
    # 建立客户端，并让用户自由选择一个在线的服务器进行连接
    client = FileClient(id)
    client.choose_server()

    if not os.path.exists(client.root_path):  # 创建数据的主文件夹
        os.mkdir(client.root_path)
    print("======================================")
    print("Welcome to FileServer %s"%(client.FS_ip+":"+client.FS_port))
    print("======================================")
    root='$ '

    while True:
        print(root,end='')
        command = input().split()
        opera = command[0].lower()
        if opera=='ls':
            filelist = client.list()
            print(filelist)

        elif opera=='lsc':  # 列出当前用户文件夹下的文件列表：
            all_files = '   '.join(os.listdir(client.root_path))  # 输出根path下的所有文件名到一个列表中
            print(all_files)

        elif opera=='upload' or opera=='u':
            target_filename = command[1]  # 要上传的目标文件名字  名字要在当前的客户数据目录下
            client.upload(target_filename)

        elif opera=='download' or opera=='d':
            target_filename = command[1]  # 要下载的目标文件名字  名字要在当前的服务器数据目录下
            client.download(target_filename)

        elif opera=='pwd':
            print(client.current_path)

        elif opera == 'delete' or command[0]=='del':
            target_filename = command[1]
            client.delete(target_filename)

        elif opera == 'mkdir':
            target_dirname = command[1]
            client.mkdir(target_dirname)

        elif opera == 'cd':
            target_dirname = command[1]
            client.cd(target_dirname)

        elif opera == 'cd..':
            client.cdb()

        elif opera == 'lso':
            client.listopen()

        elif opera == 'open':
            client.open(command[1])

        elif opera == 'close':
            client.close(command[1])

        elif opera=='help':
            client.showhelpmesg()

        else:
            print("Please enter the correct command")



if __name__ == '__main__':
    run_client(1)
    # run_client(2)