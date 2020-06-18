import grpc
import time
import os
import FileServer.FileServer_pb2 as FS_pb2, FileServer.FileServer_pb2_grpc as FS_pb2_grpc
import DirectoryServer.DirectoryServer_pb2 as DS_pb2, DirectoryServer.DirectoryServer_pb2_grpc as DS_pb2_grpc

from concurrent import futures

from setting import *


class Servicer(FS_pb2_grpc.FileServerServicer):
    def __init__(self,ip, port,id,location):
        self.id = id # 文件服务器的编号
        self.ip = ip
        self.port = port
        self.location = location
        self.root_path = '../serverData/FileServer_%s/'%(id)  # 服务器的根目录地址
        self.CHUNK_SIZE = 1024 * 1024  # 每次发送1MB的文件

        # 连接目录服务器
        dirchannel = grpc.insecure_channel(dirServer_ip+':'+dirServer_port)
        self.dirstub = DS_pb2_grpc.DirServerStub(dirchannel)
        self.online()


    def online(self):
        print("FileServer is online")
        self.dirstub.fileserver_online(DS_pb2.FileServer_On(Server_ID = self.id, ip = self.ip,port = self.port,location = self.location))

    def offline(self):
        print("FileServer is offline")
        self.dirstub.fileserver_offline(DS_pb2.FileServer_Off(Server_ID = self.id))

    def save_chunks_to_server(self,iterator):
        '''
        将文件保存到当前的文件路径上
        '''
        target_path=''
        for iter in iterator:
            target_path = iter.target_path
            with open(self.root_path+iter.target_path, 'wb') as f:
                f.write(iter.buffer)
        print(target_path)
        return target_path

    def get_file(self,path):
        '''
        根据文件名和当前路径获取文件并返回生成器
        用在客户下载申请后的返回
        '''
        with open(self.root_path+path, 'rb') as f:
            while True:
                piece = f.read(self.CHUNK_SIZE)
                if len(piece) == 0:
                    return
                yield FS_pb2.Chunk(buffer=piece)


    def get_file_chunks(self,source,target):
        '''
        用在客户上传后，服务器要把这个操作传播到每个服务器上
        注意封装成请求的时候需要封装成对方的文件地址
        source为服务器绝对地址 target为服务器相对地址
        '''
        with open(source, 'rb') as f:
            while True:
                piece = f.read(self.CHUNK_SIZE)
                if len(piece) == 0:
                    return
                yield FS_pb2.Upload_Request(buffer=piece,target_path = target)  # 同时传文件和路径


    def synchronize_upload(self,target_path):
        # 用来同步更新其他的服务器
        # 当用户上传文件的时候，需要把上传操作或者更新操作传递到所有副本上
        # 用户上传文件， 上传完之后文件服务器需要同步到所有的副本上
        source = self.root_path+target_path  # 上传文件的原地址
        target = target_path  # 上传文件的目标地址  注意这里返回的地址是相对地址
        # 这里不需要在对被更新文件上锁了，客户端已经对文件上锁
        if not os.path.exists(source):
            print("files does not exist")
            return

        # 获取所有副本服务器地址
        while True:
            try:
                response = self.dirstub.getfileserver(DS_pb2.Dir_Empty(empty=0))
                break
            except:
                print("Directory server connection timed out!")
        # 遍历所有服务器，除了自己都发一次
        for server in response.server_list:
            if server.Server_ID == self.id:
                continue

            # 建立和其他服务器的关联
            tmpchannel = grpc.insecure_channel(server.ip + ':' + server.port)
            tmpstub = FS_pb2_grpc.FileServerStub(tmpchannel)

            # 读取本地文件  注意这里返回的地址是对方的同步地址！！！
            chunks_generator = self.get_file_chunks(source, target)
            try:
                response = tmpstub.upload_without_syn(chunks_generator)
            except Exception as e:
                print(e)
                print("Server {} connection timed out!".format(server.ip+":"+server.port))
                continue
        print("File sync completed {}".format(target_path))



    def synchronize_delete(self,target_path):
        # 服务器将文件被删除的操作同步到所有副本上
        # 获取所有副本服务器地址
        while True:
            try:
                response = self.dirstub.getfileserver(DS_pb2.Dir_Empty(empty=0))
                break
            except:
                print("Directory server connection timed out!")
        for server in response.server_list:
            if server.Server_ID == self.id:
                continue
            # 对于其他服务器全部发送一次删除操作的指令

            # 建立和其他服务器的关联
            tmpchannel = grpc.insecure_channel(server.ip + ':' + server.port)
            tmpstub = FS_pb2_grpc.FileServerStub(tmpchannel)

            response = tmpstub.delete_without_syn(FS_pb2.Delete_Request(delete_path=target_path))
            if response.success == 0:
                print("File deleted successfully")
            else:
                print("File deleted failed")


    def delete_file(self,path):
        if os.path.exists(self.root_path+path):  # 如果文件存在
            os.remove(self.root_path+path)  # 则删除

    def make_directory(self,dir_path):
        if not os.path.exists(self.root_path+dir_path):  # 如果文件夹不存在 就创建文件夹
            os.mkdir(self.root_path+dir_path)




    ########################  proto 服务 ############################
    def upload_without_syn(self, request, context):
        # 用在同步中，避免重复同步 代码和update类似，但去掉同步的操作
        try:
            self.save_chunks_to_server(request)
            success=0
        except Exception as err:
            print(err)
            success=1
        return FS_pb2.Reply(success = success)


    def upload(self, request, context):
        '''
        proto服务
        提供下载接口供用户上传文件使用
        上传完后需要把文件同步到所有的文件服务器上
        '''
        try:
            target_path = self.save_chunks_to_server(request)
            self.synchronize_upload(target_path)
            success=0
        except Exception as err:
            print(err)
            success=1
        return FS_pb2.Reply(success = success)


    def download(self, request, context):
        '''
        proto服务
        提供下载接口供用户下载文件使用
        '''
        filepath = request.download_path
        if os.path.exists(self.root_path+filepath):
            return self.get_file(filepath)

    def delete(self, request, context):
        '''
        提供删除接口供用户删除文件
        '''
        try:
            self.delete_file(request.delete_path)
            self.synchronize_delete(request.delete_path)
            success=0
        except Exception as err:
            print(err)
            success=1
        return FS_pb2.Reply(success=success)


    def delete_without_syn(self, request, context):
        # 用在同步中，避免重复同步 代码和delete类似，但去掉同步的操作
        try:
            self.delete_file(request.delete_path)
            success = 0
        except Exception as err:
            print(err)
            success = 1
        return FS_pb2.Reply(success=success)


    def mkdir(self, request, context):
        '''
        提供用户建立目录的接口
        '''
        try:
            self.make_directory(request.dir_path)
            success=0
        except Exception as err:
            print(err)
            success = 1
        return FS_pb2.Reply(success=success)



    def list(self, request, context):
        all_files = '   '.join(os.listdir(self.root_path + request.cur_path))  # 输出根path下的所有文件名到一个列表中
        return FS_pb2.List_Reply(list=all_files)


    def pwd(self,request,context):
        return FS_pb2.PWD_Reply(pwd='user/') # 传一个空串 客户端只知道服务器的相对路径



def run_server(id, ip,port,location):
    servicer = Servicer(ip, port,id,location)
    if not os.path.exists(servicer.root_path):  # 创建数据的主文件夹
        os.mkdir(servicer.root_path)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    FS_pb2_grpc.add_FileServerServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{servicer.port}')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24) # 持续运行一天
    except KeyboardInterrupt:
        servicer.offline()
        server.stop(0)

if __name__=="__main__":
    # run_server(1 , 'localhost', '8001', 'LA')# 文件服务器端口号为8001
    run_server(2 , 'localhost', '8002', 'GZ')# 文件服务器端口号为8002