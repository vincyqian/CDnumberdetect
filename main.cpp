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

/******hog����********/
void coumputeHog(const Mat& src, vector<float> &descriptors)

{

	HOGDescriptor myHog = HOGDescriptor(imageSize, Size(5, 5), cvSize(3, 3), cvSize(3, 3), 9);

	myHog.compute(src.clone(), descriptors, Size(1, 1), Size(0, 0));



}

/****�����Բ������*****/
void on_mouse(int EVENT, int x, int y, int flags, void* userdata)
{
	Mat& hh = *(Mat*)userdata;
	Point2f p(x, y);

	switch (EVENT)
	{
	   case EVENT_LBUTTONDOWN://�������
	  {
		  center = Point(x, y);

	   }
	}
}

/****����Ӧ��ֵ��****/
Mat adaptbinary(Mat img)
{
	Mat imggray;
	cvtColor(img, imggray, CV_BGR2GRAY);
	Mat binary;
	int blockSize = 29; /* ��һ�η�Χ�ڣ�����19-31������Ч */
	int threshold = -10; /* ��������10֮�ڣ�ʮ��/����Ϊ����Բ��Ϊ���ҽ����ɫ */
	adaptiveThreshold(imggray, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, threshold);
	return binary;
}

/***��ȡ�ļ���****/
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

/****ѵ��*****/
int train()
{
	////��ⴰ��(128,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9  
	//HOGDescriptor hog(Size(128, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);

	////HOG���������������HOG�����ӵ�  
	//int DescriptorDim;//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������  

	//Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��  
	//Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����  
	Mat classes;
	vector<string> img_path;//�����ļ�������     
	vector<int> img_catg;
	int nLine = 0;
	string buf;

	ifstream svm_data("D:/train_list.txt");//ѵ������ͼƬ��·����д�����txt�ļ��У�ʹ��bat�������ļ����Եõ����txt�ļ�       
	ifstream svm_label("D:/label.txt");

	unsigned long n;
	while (svm_data)//��ѵ�������ļ����ζ�ȡ����      
	{
		if (getline(svm_data, buf))
		{
			nLine++;
			img_path.push_back(buf);//ͼ��·��      

		}
	}
	nLine = 0;
	while (svm_label)//��ѵ����ǩ�ļ����ζ�ȡ����      
	{
		if (getline(svm_label, buf))
		{
			nLine++;
			img_catg.push_back(atoi(buf.c_str()));//atoi���ַ���ת�������ͣ���־(0,1��2��...��9)��ע����������Ҫ��������𣬷�������      
		}
	}
	svm_data.close();//�ر��ļ� 
	svm_label.close();


	Mat data_mat, labels_mat;
	int  nImgNum = nLine; //nImgNum����������    
	cout << " ������������Ϊ�� " << nImgNum << endl;
	//data_matΪ����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��
	data_mat = Mat::zeros(nImgNum, 324, CV_32FC1);  //�С��С����ͣ��ڶ���������������������������descriptors�Ĵ�С�����ģ�������descriptors.size()�õ����Ҷ��ڲ�ͬ��С������ѵ��ͼƬ�����ֵ�ǲ�ͬ��    

													//���;���,�洢ÿ�����������ͱ�־      
													//labels_matΪѵ����������������������������������ĸ�������������1����ʱ��������޸ģ���������0����Ϊ0������1��Ϊ1
	labels_mat = Mat::zeros(nImgNum, 1, CV_32SC1);


	Mat src;
	Mat trainImg = Mat(Size(20, 20), CV_8UC3);//��Ҫ������ͼƬ������Ĭ���趨ͼƬ��28*28��С���������涨����324�����Ҫ����ͼƬ��С����������debug�鿴һ��descriptors�Ƕ��٣�Ȼ���趨��������      
	int q = img_path.size();
	//����HOG����    
	for (string::size_type i = 0; i != img_path.size(); i++)
	{
		cout << " \n�� " << i << "  ��ѭ��\n" << endl;
		src = imread(img_path[i].c_str(), 1);
		if (src.empty())
		{
			cout << " can not load the image: " << img_path[i].c_str() << endl;
			continue;
		}

		cout << " ���� " << img_path[i].c_str() << endl;

		resize(src, trainImg, trainImg.size());


		//��ⴰ��(64,128),��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9  ����Ҫ�޸�
		HOGDescriptor *hog = new HOGDescriptor(Size(20, 20), Size(10, 10), Size(5, 5), Size(5, 5), 9);
		vector<float>descriptors;//��Ž��    ΪHOG����������    
		hog->compute(trainImg, descriptors, Size(1, 1), Size(0, 0)); //Hog�������㣬��ⴰ���ƶ�����(1,1)     

																	 //cout << "HOG��������������    : " << descriptors.size() << endl;


		int    number = descriptors.size();
		//cout << "number" << number;

		//������õ�HOG�����Ӹ��Ƶ�������������data_mat  
		for (int j = 0; j < number; j++)
		{
			data_mat.at<float>(i, j) = descriptors[j];//��1�����������������еĵ�n��Ԫ��  	
		}

		labels_mat.at<int>(i, 0) = img_catg[i];
		//cout << " �������: " << img_path[i].c_str() << " " << img_catg[i] << endl;
	}

	Mat(labels_mat).copyTo(classes);
	//cout << data_mat;
	//cout << labels_mat;

	// ���������������ò���
	Ptr<SVM> SVM_params = SVM::create();
	SVM_params->setType(SVM::C_SVC);
	SVM_params->setKernel(SVM::RBF);  //�˺����������ص�����ĵط�  SVM::RBFΪ�������RBF���˺�������˹�˺�����

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
	cout << "��ʼѵ��..." << endl;

	Ptr<TrainData> traindata = ml::TrainData::create(data_mat, ROW_SAMPLE, classes);
	// ѵ��������
	SVM_params->train(traindata);
	//����ģ��
	SVM_params->save("svm.xml");
	cout << "ѵ�����ˣ�����" << endl;

	StatModel::load<SVM>("svm.xml");

	return 0;

}

/*******Ԥ��******/

int predict(Mat test)
{
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm = cv::ml::SVM::load("svm.xml");;//����ѵ���õ�xml�ļ���  
										//�������      
	char result[300]; //���Ԥ����    
	if (!test.data)
	{
		cout << "��Ԥ��ͼ�񲻴��ڣ�";
		return -1;
	}
	cv::Mat trainTempImg(cv::Size(20, 20), 8, 3);
	trainTempImg.setTo(cv::Scalar(0));
	cv::resize(test, trainTempImg, trainTempImg.size());
	cv::HOGDescriptor *hog = new cv::HOGDescriptor(cv::Size(20, 20), cv::Size(10, 10), cv::Size(5, 5), cv::Size(5, 5), 9);
	vector<float>descriptors;//�������         
	hog->compute(trainTempImg, descriptors, cv::Size(1, 1), cv::Size(0, 0));
	//cout << "HOG dims: " << descriptors.size() << endl;
	cv::Mat SVMtrainMat(1, descriptors.size(), CV_32FC1);
	int n = 0;
	for (vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++)
	{
		SVMtrainMat.at<float>(0, n) = *iter;
		n++;
	}
	int ret = svm->predict(SVMtrainMat);//�����  
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
		cout << "�������ĵ�����������ֵ"<<endl;
		imshow("img", img);
		waitKey();
	  
		cout << "�����������";
		/*****�����������������������Ϊ��ɫ*****/
		if (center.x - 210 < 0 || center.y - 210 < 0)
			exit(0);
		Mat Roi = img(Rect(center.x - 210, center.y - 210, 420, 420));
		Mat binary = adaptbinary(Roi);
		circle(binary, Point(210, 210), 120, Scalar(0), -1);
		Mat mask(Roi.rows,Roi.cols,CV_8UC1,Scalar(0));
		circle(mask, Point(210, 210), 210, Scalar(255), -1);
		binary = binary&mask;
		imshow("binary", binary);

		/*****��̬ѧ����*****/
		Mat dilateimg;
		Mat element2 = getStructuringElement(MORPH_RECT, Size(5, 5));
		dilate(binary, dilateimg, element2);  //���Ͳ���
		Mat erodeimg;
		Mat element = getStructuringElement(MORPH_RECT, Size(2, 2)); //��һ������MORPH_RECT��ʾ���εľ���ˣ���Ȼ������ѡ����Բ�εġ������͵�															
		erode(dilateimg, erodeimg, element);
		imshow("erodeimg", erodeimg);


		/*****������  Ѱ����Ӿ���*****/
		vector<vector<Point>>contours;
		findContours(erodeimg, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);//�������

		vector <vector<Point>>::iterator iter = contours.begin();
		for (; iter != contours.end();)
		{
			int cmin = 50, cmax = 500;//�ܳ���������
			double g_dConArea = contourArea(*iter);
			int c = arcLength(*iter, 1);

			if (c < 10)
			{
				iter = contours.erase(iter);//����������������ܳ�����ɸѡ
			}
			else
			{
				++iter;
			}
		}


		/*******ɸѡ��ͨ���� �и��ַ� ����ѵ������*****/
		string save_file;

		for (int i = 0; i < contours.size(); i++)
		{
			Rect rect = boundingRect(contours[i]);//���������

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






