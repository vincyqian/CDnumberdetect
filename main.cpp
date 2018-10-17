#include "opencv2/opencv.hpp"
#include<iostream>
#include <io.h>
#include <string>
#include <sstream>
#include <fstream>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace cv;
using namespace std;
using namespace cv::ml;
Point center;
Size imageSize = Size(20,20);



/****鼠标点击圆的中心*****/
void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{
	Mat& hh = *(Mat*)userdata;
	Point2f p(x, y);

	switch (EVENT)
	{
	   case EVENT_LBUTTONDOWN://按下左键
	  {
		  center = Point(x, y);

	   }
	}
}

/****自适应阈值化****/
Mat adaptbinary(Mat img)
{
	Mat imggray;
	cvtColor(img, imggray, CV_BGR2GRAY);
	Mat binary;
	int blockSize = 29; /* 在一段范围内，比如19-31，都有效 */
	int threshold = -10; /* 可在正负10之内，十字/菱形为负，圆形为正且结果反色 */
	adaptiveThreshold(imggray, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, threshold);
	return binary;
}

/***获取文件名****/
void getFiles(string path, vector<string>& files)
{
	intptr_t hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

/****训练*****/
int train()
{
	////检测窗口(128,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9  
	//HOGDescriptor hog(Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);

	////HOG检测器，用来计算HOG描述子的  
	//int DescriptorDim;//HOG描述子的维数，由图片大小、检测窗口大小、块大小、细胞单元中直方图bin个数决定  

	//Mat sampleFeatureMat;//所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数  
	//Mat sampleLabelMat;//训练样本的类别向量，行数等于所有样本的个数，列数等于1；1表示有人，-1表示无人  
	Mat classes;
	vector<string> img_path;//输入文件名变量     
	vector<int> img_catg;
	int nLine = 0;
	string buf;

	ifstream svm_data("D:/train_list.txt");//训练样本图片的路径都写在这个txt文件中，使用bat批处理文件可以得到这个txt文件       
	ifstream svm_label("D:/label.txt");

	unsigned long n;
	while (svm_data)//将训练样本文件依次读取进来      
	{
		if (getline(svm_data, buf))
		{
			nLine++;
			img_path.push_back(buf);//图像路径      

		}
	}
	nLine = 0;
	while (svm_label)//将训练标签文件依次读取进来      
	{
		if (getline(svm_label, buf))
		{
			nLine++;
			img_catg.push_back(atoi(buf.c_str()));//atoi将字符串转换成整型，标志(0,1，2，...，9)，注意这里至少要有两个类别，否则会出错      
		}
	}
	svm_data.close();//关闭文件 
	svm_label.close();


	Mat data_mat, labels_mat;
	int  nImgNum = nLine; //nImgNum是样本数量    
	cout << " 共有样本个数为： " << nImgNum << endl;
	//data_mat为所有训练样本的特征向量组成的矩阵，行数等于所有样本的个数，列数等于HOG描述子维数
	data_mat = Mat::zeros(nImgNum, 324, CV_32FC1);  //行、列、类型；第二个参数，即矩阵的列是由下面的descriptors的大小决定的，可以由descriptors.size()得到，且对于不同大小的输入训练图片，这个值是不同的    

													//类型矩阵,存储每个样本的类型标志      
													//labels_mat为训练样本的类别向量，行数等于所有样本的个数，列数等于1；暂时，后面会修改，比如样本0，就为0，样本1就为1
	labels_mat = Mat::zeros(nImgNum, 1, CV_32SC1);


	Mat src;
	Mat trainImg = Mat(Size(20, 20), CV_8UC3);//需要分析的图片，这里默认设定图片是28*28大小，所以上面定义了324，如果要更改图片大小，可以先用debug查看一下descriptors是多少，然后设定好再运行      
	int q = img_path.size();
	//处理HOG特征    
	for (string::size_type i = 0; i != img_path.size(); i++)
	{
		cout << " \n第 " << i << "  次循环\n" << endl;
		src = imread(img_path[i].c_str(), 1);
		if (src.empty())
		{
			cout << " can not load the image: " << img_path[i].c_str() << endl;
			continue;
		}

		cout << " 处理： " << img_path[i].c_str() << endl;

		resize(src, trainImg, trainImg.size());


		//检测窗口(64,128),块尺寸(16,16),块步长(8,8),cell尺寸(8,8),直方图bin个数9  ，需要修改
		HOGDescriptor *hog = new HOGDescriptor(Size(20, 20), Size(10, 10), Size(5, 5), Size(5, 5), 9);
		vector<float>descriptors;//存放结果    为HOG描述子向量    
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //Hog特征计算，检测窗口移动步长(1,1)     

																	 //cout << "HOG描述子向量个数    : " << descriptors.size() << endl;


		int    number = descriptors.size();
		//cout << "number" << number;

		//将计算好的HOG描述子复制到样本特征矩阵data_mat  
		for (int j = 0; j < number; j++)
		{
			data_mat.at<float>(i, j) = descriptors[j];//第1个样本的特征向量中的第n个元素  	
		}

		labels_mat.at<int>(i, 0) = img_catg[i];
		//cout << " 处理完毕: " << img_path[i].c_str() << " " << img_catg[i] << endl;
	}

	Mat(labels_mat).copyTo(classes);
	//cout << data_mat;
	//cout << labels_mat;

	// 创建分类器并设置参数
	Ptr<SVM> SVM_params = SVM::create();
	SVM_params->setType(SVM::C_SVC);
	SVM_params->setKernel(SVM::RBF);  //核函数，后期重点分析的地方  SVM::RBF为径向基（RBF）核函数（高斯核函数）

	SVM_params->setDegree(10.0);
	SVM_params->setGamma(0.09);
	SVM_params->setCoef0(1.0);
	SVM_params->setC(10.0);
	SVM_params->setNu(0.5);
	SVM_params->setP(1.0);
	SVM_params->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 1000, 0.01));


	Mat labelMat1(labels_mat.rows, labels_mat.cols, CV_32SC1);
	for (int i = 0; i < labels_mat.rows; i++)
	{
		for (int j = 0; j < labels_mat.cols; j++)
		{
			labelMat1.at<int>(i, j) = labels_mat.at<float>(i, j);
		}
	}
	cout << "开始训练..." << endl;

	Ptr<TrainData> traindata = ml::TrainData::create(data_mat, ROW_SAMPLE, classes);
	// 训练分类器
	SVM_params->train(traindata);
	//保存模型
	SVM_params->save("svm.xml");
	cout << "训练好了！！！" << endl;

	StatModel::load<SVM>("svm.xml");

	return 0;

}

/*******预测******/

int predict(Mat test)
{
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm = cv::ml::SVM::load("svm.xml");;//加载训练好的xml文件，  
										//检测样本      
	char result[300]; //存放预测结果    
	if (!test.data)
	{
		cout << "待预测图像不存在！";
		return -1;
	}
	cv::Mat trainTempImg(cv::Size(20, 20), 8, 3);
	trainTempImg.setTo(cv::Scalar(0));
	cv::resize(test, trainTempImg, trainTempImg.size());
	cv::HOGDescriptor *hog = new cv::HOGDescriptor(cv::Size(20, 20), cv::Size(10, 10), cv::Size(5, 5), cv::Size(5, 5), 9);
	vector<float>descriptors;//结果数组         
	hog->compute(trainTempImg, descriptors, cv::Size(1, 1), cv::Size(0, 0));
	//cout << "HOG dims: " << descriptors.size() << endl;
	cv::Mat SVMtrainMat(1, descriptors.size(), CV_32FC1);
	int n = 0;
	for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
	{
		SVMtrainMat.at<float>(0, n) = *iter;
		n++;
	}
	int ret = svm->predict(SVMtrainMat);//检测结果  
	return ret;
}

int main(int argc, char ** argv)
{
	if (argc != 2)
	{
		cout << "usage : argv[0] imagepath" << endl;
		return 1;
	}

	string filedir = argv[1];
	vector<string> files;
	getFiles(filedir, files);
	int img_index = 0;
	ofstream outfile("D:/train_list.txt");

	for (vector<string>::iterator i = files.begin(); i != files.end(); i++)
	{
		if ((*i).find("bmp") == -1)//filter by .ext
			continue;
		cout << "processing : " << i->c_str() << endl;
		Mat img = imread((*i));

			outfile << i->c_str();
			outfile << "  ";
			outfile << "0";
			outfile << endl;

		resize(img, img, Size(0, 0), 0.5, 0.5);
		namedWindow("img", WINDOW_NORMAL);
		setMouseCallback("img", on_mouse, &img);
		cout << "请点击中心点后输入任意键值"<<endl;
		imshow("img", img);
		waitKey();
	  
		cout << "请输入任意键";
		/*****将环形区域以外的区域设置为黑色*****/
		if (center.x - 210 < 0 || center.y - 210 < 0)
			exit(0);
		Mat Roi = img(Rect(center.x - 210, center.y - 210, 420, 420));
		Mat binary = adaptbinary(Roi);
		circle(binary, Point(210, 210), 120, Scalar(0), -1);
		Mat mask(Roi.rows,Roi.cols,CV_8UC1,Scalar(0));
		circle(mask, Point(210, 210), 210, Scalar(255), -1);
		binary = binary&mask;
		imshow("binary", binary);

		/*****形态学操作*****/
		Mat dilateimg;
		Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));
		dilate(binary, dilateimg, element2);  //膨胀操作
		Mat erodeimg;
		Mat element = getStructuringElement(MORPH_RECT, Size(2, 2)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的															
		erode(dilateimg, erodeimg, element);
		imshow("erodeimg", erodeimg);


		/*****区域检测  寻找外接矩形*****/
		vector<vector<Point>>contours;
		findContours(erodeimg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//轮廓检测

		vector <vector<Point>>::iterator iter = contours.begin();
		for (; iter != contours.end();)
		{
			int cmin = 50, cmax = 500;//周长的上下限
			double g_dConArea = contourArea(*iter);
			int c = arcLength(*iter, 1);

			if (c < 10)
			{
				iter = contours.erase(iter);//对轮廓按照面积、周长进行筛选
			}
			else
			{
				++iter;
			}
		}


		/*******筛选连通区域 切割字符 生成训练数据*****/
		string save_file;

		for (int i = 0; i < contours.size(); i++)
		{
			Rect rect = boundingRect(contours[i]);//检测外轮廓

			if (rect.width < 35 && rect.width>8 && rect.height < 35 && rect.height>8 && rect.x + 0.5*rect.width - 10 > 0 &&
				rect.y + 0.5*rect.height - 10 > 0 && rect.x + 0.5*rect.width + 10 < 420 && rect.y + 0.5*rect.height + 10 < 420)
			{


				Rect train = Rect(rect.x + 0.5*rect.width - 10, rect.y + 0.5*rect.height - 10, 20, 20); 
				int pre = predict(Roi(train));
				if (pre == -1)
				{
					rectangle(binary, train, Scalar(255), 3);
				}
				//save_file = "D:/train/" + to_string(img_index) + ".bmp";
				//imwrite(save_file, Roi(train));
				img_index++;
			}
			}
			imshow("binary", binary);
			waitKey();
			destroyAllWindows();


		}
	}






