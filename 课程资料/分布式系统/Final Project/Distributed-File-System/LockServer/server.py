# 锁服务器，用来实现对文件的上锁防止被同时写
import grpc
import time
import os

import LockServer.LockServer_pb2 as LS_pb2, LockServer.LockServer_pb2_grpc as LS_pb2_grpc

from concurrent import futures

from setting import *

class Servicer(LS_pb2_grpc.LockServerServicer):
    def __init__(self):
        self.ip = lockServer_ip
        self.port = lockServer_port
        self.root_path = '../serverData/LockServer/'
        self.lockfile_name = 'lockFile.txt'
        self.lockfile_path = self.root_path + self.lockfile_name
        self.online()

    def online(self):
        print("LockServer is online")

    def offline(self):
        print("LockServer is offline")

    def get_Lock_list(self):
        # 获取所有文件锁
        f = open(self.lockfile_path, 'r', encoding='utf-8')
        lock_dic = {}  # 嵌套字典 key是lock的id
        for line in f.readlines():
            tmp = line.strip().split()
            if tmp != '':
                lock_dic[tmp[0]+tmp[1]] = {'filepath':tmp[0],'filename':tmp[1],'client_id': tmp[2]}  # 为方便使用文件路径名作为索引
        f.close()
        return lock_dic



    def addlock(self,filepath,filename,client_id):
        lock_dic = self.get_Lock_list()
        if filepath+filename in lock_dic.keys(): # 如果文件锁已经存在 则不允许上锁
            ex = Exception("File is locked ==> "+filepath+":"+filename)
            # 抛出异常对象
            raise ex

        # 新增新插入锁
        f = open(self.lockfile_path, 'a', encoding='utf-8')
        text = str(filepath) + '\t' + str(filename) + '\t' + str(client_id)+'\n'
        f.writelines(text)

        print("New file lock ==> ", "filepath: {}   filename: {}   client_id: {} ".format(filepath, filename, client_id))
        f.close()



    def unlock(self,filepath,filename):
        # 根据文件名字删除对应文件锁
        server_dic = self.get_Lock_list()
        server_dic.pop(filepath+filename)
        print("file unlock ==> ", "filepath: {}   filename: {} ".format(filepath, filename))
        with open(self.lockfile_path, 'w', encoding='utf-8') as f:
            for id in sorted(server_dic):
                text = str(server_dic[id]['filepath']) + '\t' + str(server_dic[id]['filename']) + '\t' + str(
                    server_dic[id]['client_id']) + '\n'  # 将删除后的目录重新写入文件
                f.writelines(text)



    ########################  proto 服务 ############################
    def lockfile(self, request, context):
        try:
            self.addlock(request.file_path,request.filename,request.client_id)
            success=0
        except Exception as e:
            print(e)
            success=1
        return LS_pb2.Lock_Reply(success=success)


    def unlockfile(self, request, context):
        try:
            self.unlock(request.file_path,request.filename)
            success = 0
        except Exception as e:
            print(e)
            success = 1
        return LS_pb2.Lock_Reply(success=success)



def run_server():
    servicer = Servicer()
    if not os.path.exists(servicer.root_path):  # 创建数据的主文件夹
        os.mkdir(servicer.root_path)
        open(servicer.lockfile_path, 'w', encoding='utf-8')  # 创建目标文件

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    LS_pb2_grpc.add_LockServerServicer_to_server(servicer, server)
    server.add_insecure_port(f'[::]:{servicer.port}')
    server.start()
    try:
        while True:
            time.sleep(60 * 60 * 24)  # 持续运行一天
    except KeyboardInterrupt:
        servicer.offline()
        server.stop(0)


if __name__=="__main__":
    run_server()