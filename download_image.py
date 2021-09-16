import pymysql, os, json, shutil, requests, cv2
from os.path import join



def main1():
	bbb = '/data/wuxiaopeng/datasets/tmp2'
	conn = pymysql.connect(host='rm-uf601w4i44vm67jg0.mysql.rds.aliyuncs.com', port=3306, user='fmadmin',
	                       passwd='1q2w#E$R', db='kf_patrol_huidu', charset='utf8')
	cur = conn.cursor()
	sql = "select oss_image_url from publish_design"
	cur.execute(sql)
	urls = cur.fetchall()
	name = 40000
	print(name)
	for numi, i in enumerate(urls):
		i = i[0].split(",")[0]
		if "sample" in i:
			i = i.replace("sample", "origin")
		if "_小样" in i:
			tmp= i.find("_小样")
			i = i[:tmp]+i[tmp+3:]
		# print(i)
		path = os.path.join(bbb, str(name)+ ".jpg")
		while os.path.exists(path):
			name+=1
			path = os.path.join(bbb, str(name) + ".jpg")
		with open(path,"wb") as f:
			try:

				response = requests.get(i, timeout = 3)
				f.write(response.content)
				# print(numi,i)
			except:

				print("error url ", numi,i,)
				try:
					shutil.rmtree(path)
				except:
					pass


		# if numi==100:
		# 	break

def main2():
	root = "/data/wuxiaopeng/datasets/compare_2_picture/centernet/train_generate/generate/label_final"
	bbb = '/data/wuxiaopeng/datasets/detect_inverse_image'
	print(len("/data/wuxiaopeng/datasets/compare_2_picture/"))
	name = 1
	data = []
	data_label = []
	for i in os.listdir(root):
		json_path = join(root, i)

		with open(json_path, "r") as f:
			json_data = json.load(f)

			path = json_data["path"][:44] + "centernet/" + json_data["path"][44:]
			if os.path.exists(path):
				path2 = os.path.join(bbb, str(name)+".jpg")
				while os.path.exists(path2):
					name+=1
					path2 = os.path.join(bbb, str(name) + ".jpg")
				shutil.copy(path, path2)

def main3():
	#检查图片的正确性，判断图片否为空
	bbb = '/data/wuxiaopeng/datasets/tmp2'
	for numi,i in enumerate(os.listdir(bbb)):
		path = join(bbb,i)
		img = cv2.imread(path)
		try:
			a = len(img)
			print(numi,a)
		except Exception as e:
			print(e)
			os.remove(path)

def main4():
	# 提前resize好图片
	aaa = '/data/wuxiaopeng/datasets/tmp1'
	bbb = '/data/wuxiaopeng/datasets/tmp2'
	for numi, i in enumerate(os.listdir(bbb)):
		print(numi,i)
		path = join(bbb, i)
		path2 = join(aaa,i)
		img = cv2.imread(path)
		img = cv2.resize(img,(224,224), interpolation=cv2.INTER_LINEAR)
		cv2.imwrite(path2, img)



if __name__ == '__main__':
	main4()



