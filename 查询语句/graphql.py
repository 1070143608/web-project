import requests
import os


base_dir =  os.path.dirname(os.path.abspath('__file__'))
MEDIA_URL = "http://testdev:9083/media/graphiql/"
SHOP_URL = "http://testdev:9091/shop/graphiql/"
DOWNLOAD_DIR = os.path.join(base_dir,'download')


class Graphql:
    def __init__(self, media_url=MEDIA_URL, shop_url=SHOP_URL,
                 download_dir=DOWNLOAD_DIR):
        self.media_url = media_url
        self.shop_url = shop_url
        self.download_dir = download_dir

    # 获取SPU下载地址，用于下载文件.返回一个字典，键为sku的id，键值为对应的下载地址
    def query_spu_download_message(self, spu_id):
        print("Querying original GLBs of SPU...")
        data = {'query': '''{
              spu(id:"''' + spu_id + '''"){
            threedModel {
              edges {
                node {
                  id
                  uploadedModelMediamsFileId
                  uploadedModelMediamsFileIdInfo
                  displayModelMediamsFileId
                }
              }
            }
          }
        }'''}
        upload_response = requests.post(url=self.shop_url, data=data).json()
        dic = {}
        for node in upload_response['data']['spu']["threedModel"]["edges"]:
            node = node["node"]
            upload_id = node['id']
            for key, value in node['uploadedModelMediamsFileIdInfo'].items():
                upload_info = value[0]
                dic[upload_id] = upload_info["cdn_host"] + '/' + upload_info["key"]
        print("Find original GLBs successfully")
        return dic

    # 下载sku glb格式文件, 返回下载文件的ids
    def download_glb(self, spu_id):
        upload_ids = []
        for upload_id, upload_info in self.query_spu_download_message(spu_id).items():
            media = requests.get(upload_info)
            if media.status_code == 200:
                print("Start downloading original GLBs...")
                glb_path = os.path.join(self.download_dir, upload_id + '.glb')
                with open(glb_path, 'wb') as f:
                    f.write(media.content)
                upload_ids.append(upload_id)
                print("Download all GLBs successfully")
            else:
                print("Fail to download GLB")
        return upload_ids

    # 查询SPU信息
    def query_spu(self, spu_id):
        data = {'query': '''{
        products(spu_Id:"''' + spu_id + '''") {
          edges {
            node {
              id
              productSelectValues(modelMesh_Isnull:false){
                edges {
                  node {
                    modelMesh {
                        nameCode
                    }
                    selectValue{
                        value
                    }
                  }
                }
              }
            }
          }
        }
      }'''}
        response = requests.post(url=self.shop_url, data=data)
        return response

    # 处理SPU字典信息
    def process_spu(self, spu_id):
        data = self.query_spu(spu_id).json()
        dic = {}
        for sku in data["data"]["products"]["edges"]:
            sku_id = sku["node"]["id"]
            dic[sku_id] = {}
            for node in sku["node"]["productSelectValues"]["edges"]:
                mesh = node["node"]
                mesh_name = mesh["modelMesh"]["nameCode"]
                if not dic[sku_id].get(mesh_name):
                    dic[sku_id][mesh_name] = {}
                attr_value = mesh["selectValue"]["value"]
                for key, value in attr_value.items():
                    dic[sku_id][mesh_name][key] = value
        return dic

    # 获取上传文件信息 media
    def get_media_upload_path(self):
        data = {'query': '''{
          resources(
            service: "shop", 
            attribute:"display_model_mediams_file_id",
            table:"shop_threed_spu_product_3d_models"
            ){
            edges{
              node{
                id
                attribute
                presignedPostUrl(count:1){
                  url
                  fields
                }
                mediaType {
                  type
                  mime
                  suffix
                }
                maxFilesize
                minFilesize
              }
            }
            serverStatus {
              code
              message
            }
          }
        }'''}
        media = requests.post(self.media_url, data=data)
        if media.status_code == 200:
            print("Successfully get authorization.")
        else:
            print("Fail to get authorization.")
            return 0
        response = media.json()
        fields = response['data']['resources']['edges'][0]['node']['presignedPostUrl'][0]['fields']
        key = fields['key']
        url = response['data']['resources']['edges'][0]['node']['presignedPostUrl'][0]['url']
        return fields, key, url

    # 绑定文件
    def bond(self, upload_id, key):
        data = {'query': '''mutation{
            updateSpuProduct3dModel(input:{
             id:"''' + upload_id + '''",
             displayModelMediamsFileId:"''' + key + '''"
            }){
             serverStatus {
               code
               message
             }
            }
            }'''}
        shop = requests.post(self.shop_url, data=data)
        response = shop.json()
        status_code = response['data']['updateSpuProduct3dModel']['serverStatus']['code']
        if status_code == 200:
            print("Successfully bond file to spu.")
        else:
            print("Fail to bond.")

    # 上传文件
    def upload(self, file_path, upload_id):
        fields, key, url = self.get_media_upload_path()
        body = fields
        file = open(file_path, 'rb')
        files = {'file': file}
        upload = requests.post(url, data=body, files=files)
        if upload.status_code == 201:
            print("Successfully upload file.")
        else:
            print("Fail to upload.")
        self.bond(upload_id, key)
        return 0
