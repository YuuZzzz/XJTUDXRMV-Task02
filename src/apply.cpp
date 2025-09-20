#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class LightDescriptor {
public:
    RotatedRect rotated_rect;
    float length;
    float width;
    Point2f center;
    float angle;

    LightDescriptor() {}
    LightDescriptor(const RotatedRect& rRect) {
        rotated_rect = rRect;
        center = rRect.center;
        if (rRect.size.width > rRect.size.height) {
            length = rRect.size.width;
            width = rRect.size.height;
            angle = 90 - abs(rRect.angle);
        } else {
            length = rRect.size.height;
            width = rRect.size.width;
            angle = abs(rRect.angle);
        }
    }
};

class ArmorDescriptor {
public:
    LightDescriptor light_bars[2];
    Rect bounding_rect;

    ArmorDescriptor() {}
    ArmorDescriptor(const LightDescriptor& l1, const LightDescriptor& l2) {
        if (l1.center.x < l2.center.x) {
            light_bars[0] = l1;
            light_bars[1] = l2;
        } else {
            light_bars[0] = l2;
            light_bars[1] = l1;
        }
        Point2f left_pts[4], right_pts[4];
        light_bars[0].rotated_rect.points(left_pts);
        light_bars[1].rotated_rect.points(right_pts);
        vector<Point2f> hull_pts;
        for(int i=0; i<4; ++i) {
            hull_pts.push_back(left_pts[i]);
            hull_pts.push_back(right_pts[i]);
        }
        bounding_rect = boundingRect(hull_pts);
    }
};

void detectArmorPlate() {

    //预设参数
    const double LIGHT_MIN_AREA = 15.0;
    const double LIGHT_MAX_ASPECT_RATIO = 0.5;
    const double LIGHT_MAX_ANGLE = 25.0;
    const double ARMOR_MAX_ANGLE_DIFF = 7.0;
    const double ARMOR_MAX_LENGTH_RATIO = 0.3;
    const double ARMOR_MAX_Y_DIFF_RATIO = 0.5;
    const double ARMOR_MIN_X_DIFF_RATIO = 0.5;
    const double ARMOR_MAX_X_DIFF_RATIO = 4.0;

    //预处理
    Mat img = imread("../resources/test_image_2.png");
    if (img.empty()) { cout << "无法读取图像" << endl; return; }
    imshow("原图", img); 
    waitKey(0);

    vector<Mat> channels;
    split(img, channels);
    Mat gray = channels[0] - channels[2];
    imshow("通道相减(B-R)", gray); 
    waitKey(0);

    //第一次阈值化，提取亮环
    //目标是只保留亮的轮廓，压制所有灰色背景
    Mat ring_binary;
    threshold(gray, ring_binary, 183,255, THRESH_BINARY); //fine tune
    imshow("二值化", ring_binary); 
    waitKey(0);

    //反相，得到灯条
    Mat inverted_binary;
    bitwise_not(ring_binary, inverted_binary);
    imshow("反相", inverted_binary); 
    waitKey(0);

    //寻找轮廓并根据层级和面积筛选
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    //使用RETR_CCOMP得到两层轮廓，外层是-1，内层指向外层
    findContours(inverted_binary, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    vector<LightDescriptor> light_infos;
    for (int i = 0; i >= 0; i = hierarchy[i][0]) { // 遍历所有外层轮廓
        // 找到的第一个轮廓通常是图像边框，面积最大，跳过
        if (i == 0 && contours.size() > 1) continue; 

        // 遍历该外层轮廓的所有内层轮廓（灯条）
        for (int j = hierarchy[i][2]; j >= 0; j = hierarchy[j][0]) {
            if (contourArea(contours[j]) < LIGHT_MIN_AREA) continue;

            RotatedRect lightRec = minAreaRect(contours[j]);
            LightDescriptor light(lightRec);

            if (light.width / light.length > LIGHT_MAX_ASPECT_RATIO ||
                light.angle > LIGHT_MAX_ANGLE) {
                continue;
            }
            light_infos.push_back(light);
        }
    }

    //可视化与匹配
    Mat light_bars_vis = img.clone();
    cout << "找到了 " << light_infos.size() << " 个灯条" << endl;
    for (const auto& light : light_infos) {
        Point2f vertices[4];
        light.rotated_rect.points(vertices);
        for (int k = 0; k < 4; k++) line(light_bars_vis, vertices[k], vertices[(k + 1) % 4], Scalar(255, 0, 0), 2);
    }
    imshow("筛选出的灯条", light_bars_vis); 
    waitKey(0);

    //匹配
    if (light_infos.size() < 2) { cout << "灯条数量不足" << endl; destroyAllWindows(); return; }
    vector<ArmorDescriptor> armors;
    sort(light_infos.begin(), light_infos.end(), [](const LightDescriptor& l1, const LightDescriptor& l2) { return l1.center.x < l2.center.x; });
    for (size_t i = 0; i < light_infos.size() - 1; i++) {
        for (size_t j = i + 1; j < light_infos.size(); j++) {
            const LightDescriptor& leftLight = light_infos[i]; const LightDescriptor& rightLight = light_infos[j];
            float angleDiff = abs(leftLight.angle - rightLight.angle);
            float lenDiff_ratio = abs(leftLight.length - rightLight.length) / max(leftLight.length, rightLight.length);
            if (angleDiff > ARMOR_MAX_ANGLE_DIFF || lenDiff_ratio > ARMOR_MAX_LENGTH_RATIO) continue;
            float meanLen = (leftLight.length + rightLight.length) / 2;
            float yDiff_ratio = abs(leftLight.center.y - rightLight.center.y) / meanLen;
            float xDiff_ratio = abs(leftLight.center.x - rightLight.center.x) / meanLen;
            if (yDiff_ratio > ARMOR_MAX_Y_DIFF_RATIO || xDiff_ratio < ARMOR_MIN_X_DIFF_RATIO || xDiff_ratio > ARMOR_MAX_X_DIFF_RATIO) continue;
            armors.emplace_back(leftLight, rightLight);
        }
    }
    
    //绘制结果
    Mat result_img = img.clone();
    cout << "最终匹配到 " << armors.size() << " 个装甲板" << endl;
    for (const auto& armor : armors) {
        rectangle(result_img, armor.bounding_rect, Scalar(0, 255, 0), 2);
    }
    imshow("识别结果", result_img);
    waitKey(0);
    destroyAllWindows();
}


int main() {
    detectArmorPlate();
    destroyAllWindows();
    return 0;
}