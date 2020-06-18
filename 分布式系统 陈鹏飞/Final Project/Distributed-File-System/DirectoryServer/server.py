import grpc
import time
import os

import DirectoryServer.DirectoryServer_pb2 as DS_pb2, DirectoryServer.DirectoryServer_pb2_grpc as DS_pb2_grpc

from concurrent import futures

from setting import *

class DirServicer(DS_pb2_grpc.DirServerServicer):
    def __init__(self):
        # 需要先启动目录服务器
        self.ip = dirServer_ip
        self.port = dirServer_port
        self.root_path = '../serverData/DirServer/' # 定义目录服务器的根目录
        self.dirFile_name = 'dirFile.txt' # 定义目录服务器目录文件
        self.path = self.root_path + self.dirFile_name
        self.online()

    def online(self):
        print("DirServer is online")

    def offline(self):
        print("DirServer is offline")

    def get_Server_list(self):
        f = open(self.path, 'r', encoding='utf-8')
        server_dic={}  # 嵌套字典，key是id values是ip和port的字典
        for line in f.readlines():
            tmp=line.strip().split()
            if tmp != '':
                server_dic[int(tmp[0])]={'ip':tmp[1],'port':tmp[2],'location':tmp[3]}
        f.close()
        return server_dic


    def addServer(self,id,ip,port,location):
        # 新增文件服务器进入目录中  如果服务器已经存在就不再重复插入
        server_dic = self.get_Server_list()
        f = open(self.path, 'a', encoding = 'utf-8')

        if not id in server_dic.keys():
            text = str(id)+'\t'+str(ip)+'\t'+str(port)+'\t'+str(location)+'\n'
            f.writelines(text)
        print("New FileServer login ==> ","ID: {}   ip: {}   port: {}   location: {}".format(id,ip,port,location))
        f.close()


    def delServer(self, id):
        # 根据id删除对应服务器
        server_dic = self.get_Server_list()
        server_dic.pop(id)
        with open(self.path,'w',encoding='utf-8') as f:
            for id in sorted(server_dic):
                text = str(id)+'\t'+str(server_dic[id]['ip'])+'\t'+str(server_dic[id]['port'])+'\t'+str(server_dic[id]['location'])+'\n'  # 将删除后的目录重新写入文件
                f.writelines(text)


    ########################  proto 服务 ############################
    def fileserver_online(self, request, context):
        # 服务器上线，新增目录记录
        try:
            self.addServer(request.Server_ID,request.ip,request.port,request.location)
            success=0
        except Exception as e:
            print(e)
            success=1
        return DS_pb2.Dir_Reply(success=success)



    def fileserver_offline(self, request, context):
        # 服务器下线，去掉目录记录
        try:
            self.delServer(request.Server_ID)
            success = 0
        except Exception as e:
            print(e)
            success = 1
        return DS_pb2.Dir_Reply(success=success)

    def getfileserver(self, request,context):
        # 获取所有文件服务器
        server_dic = self.get_Server_list()  # 将所有的服务器按照id大小排序展示
        server_list=[]
        for id in sorted(server_dic):
            server_list.append(DS_pb2.FileServer_info(Server_ID=id, ip =server_dic[id]['ip'],
                                               port =server_dic[id]['port'],location =server_dic[id]['location']))

        return DS_pb2.FileServer_List(server_list=server_list)



def run_server():
    servicer = DirServicer()
    if not os.path.exists(servicer.root_path):  # 创建数据的主文件夹
        os.mkdir(servicer.root_path)
        open(servicer.path, 'w', encoding='utf-8') # 创建目标文件

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    DS_pb2_grpc.add_DirServerServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{servicer.port}')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)  # 持续运行一天
    except KeyboardInterrupt:
        servicer.offline()
        server.stop(0)


if __name__=="__main__":
    run_server()# 目录服务器端口号固定位8000