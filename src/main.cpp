#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std; 

int main() {
    //读取原始图像
    Mat img = imread("../resources/test_image.png");
    if (img.empty()) {
        cout << "无法读取图像" << endl;
        return -1;
    }
    imshow("原图", img);
    waitKey(0);

    // 1. 图像颜色空间转换
    // a. 转化为灰度图
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    imshow("灰度图", gray_img);
    waitKey(0);

    // b. 转化为 HSV 图片
    Mat hsv_img;
    cvtColor(img, hsv_img, COLOR_BGR2HSV);
    imshow("HSV图", hsv_img);
    waitKey(0);

    // 2. 应用各种滤波操作
    // a. 应用均值滤波
    Mat blur_img;
    blur(img, blur_img, Size(7, 7));
    imshow("均值滤波", blur_img);
    waitKey(0);

    // b. 应用高斯滤波
    Mat gaussian_img;
    GaussianBlur(img, gaussian_img, Size(7, 7), 0);
    imshow("高斯滤波", gaussian_img);
    waitKey(0);

    // 3. 特征提取
   // a. HSV提取红色颜色区域
    Mat red_mask;
    Mat mask1, mask2;
    inRange(hsv_img, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
    inRange(hsv_img, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);
    red_mask = mask1 | mask2;
    // 显示红色区域掩码
    imshow("提取红色区域掩码", red_mask);
    waitKey(0);
    // b. 寻找图像中红色的外轮廓
    vector<vector<Point>> contours;
    findContours(red_mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // c. 计算面积, 并准备用于绘制的图像副本
    Mat contours_bbox_drawing = img.clone(); // 创建副本用于绘制轮廓和BBox
    cout << "找到 " << contours.size() << " 个红色轮廓。" << endl;
    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area < 100) continue; // 过滤噪声
        cout << "轮廓 #" << i << " 的面积: " << area << endl;
        // 绘制红色的外轮廓 (用绿色绘制)
        drawContours(contours_bbox_drawing, contours, (int)i, Scalar(0, 255, 0), 2);
        // 绘制红色的 Bounding Box (用蓝色绘制)
        Rect bbox = boundingRect(contours[i]);
        rectangle(contours_bbox_drawing, bbox, Scalar(255, 0, 0), 2);
    }
    // 显示绘制了轮廓和BoundingBox的图像
    imshow("红色区域轮廓和BoundingBox", contours_bbox_drawing);
    waitKey(0);



     // d. 提取高亮颜色区域并进行图形学处理
    Mat bright_mask, bright_region;
    // d1. 提取高亮区域 (灰度值大于200)
    inRange(gray_img, Scalar(200), Scalar(255), bright_mask);
    img.copyTo(bright_region, bright_mask); // 将原图的高亮部分拷贝出来
    imshow("高亮区域", bright_region);
    waitKey(0);
    // d2. 灰度化
    Mat bright_gray;
    cvtColor(bright_region, bright_gray, COLOR_BGR2GRAY);
    imshow("灰度化", bright_gray);
    waitKey(0); 
    // d3. 二值化
    Mat bright_binary;
    threshold(bright_gray, bright_binary, 1, 255, THRESH_BINARY);
    imshow("二值化", bright_binary);
    waitKey(0);
    // d4. 膨胀和腐蚀
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat bright_dilated, bright_eroded;
    dilate(bright_binary, bright_dilated, kernel);
    erode(bright_dilated, bright_eroded, kernel); // 先膨胀再腐蚀（闭运算）来填充小洞
    imshow("先膨胀再腐蚀", bright_eroded);
    waitKey(0);
    // d5. 对处理后的图像进行漫水处理 (Flood Fill)
    Mat floodfill_img = bright_eroded.clone();
    // 在(0,0)点开始填充，如果(0,0)是黑色(0)，则将其和所有相连的黑色区域变成灰色(128)
    // 漫水填充通常用于填充闭合区域内部
    floodFill(floodfill_img, Point(0, 0), Scalar(128));
    imshow("漫水处理", floodfill_img);
    waitKey(0);
  



    // 4. 图像绘制
    Mat drawing_board = img.clone();
    // a. 绘制任意圆形方形和文字
    circle(drawing_board, Point(150, 150), 80, Scalar(0, 0, 255), 3); // 红色圆形
    rectangle(drawing_board, Point(300, 100), Point(500, 300), Scalar(255, 255, 0), -1); // 青色实心矩形
    putText(drawing_board, "ZJY opencv display", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.5, Scalar(0, 255, 255), 3); // 黄色文字
    imshow("图像绘制", drawing_board);
    waitKey(0);

    // 5. 对图像进行处理
    // a. 图像旋转 35 度
    Mat rotated_img;
    Point2f center(img.cols / 2.0, img.rows / 2.0); // 定义旋转中心
    Mat rot = getRotationMatrix2D(center, 35, 1.0); // 获取旋转矩阵 (角度35, 缩放1.0)
    warpAffine(img, rotated_img, rot, img.size()); // 应用仿射变换
    imshow("旋转35度", rotated_img);
    waitKey(0);

    // b. 图像裁剪为原图的左上角 1/4
    // 定义一个 Rect 区域: (x, y, width, height)
    Rect crop_region(0, 0, img.cols / 2, img.rows / 2);
    Mat cropped_img = img(crop_region);
    imshow("裁剪左上角", cropped_img);
    waitKey(0);

    //结束
    cout << "操作完毕" << endl;
    destroyAllWindows(); // 关闭所有打开的窗口
    return 0;
}