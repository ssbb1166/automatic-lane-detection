#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;


// 프레임 변수
Mat frame, gray, edges, result;
int width, height;


// HoughLinesP 함수 인자
double rho = 2;
double theta = CV_PI / 180;
int hough_threshold = 100;
double minLineLength = 100;
double maxLineGap = 250;


// 기울기 threshold
double slope_min_threshold = 0.65;
double slope_max_threshold = 1.2;


// 누적 line 저장 벡터 변수
vector<Vec4i> r_save_lines;
int right_sum[4] = { 0 };
int right_avg[4];
int r_save_cnt = 0;
vector<Vec4i> l_save_lines;
int left_sum[4] = { 0 };
int left_avg[4];
int l_save_cnt = 0;


// 화면에서 line을 검출할 영역(ROI) 설정
Mat setROI(Mat edges, Point* points)
{
	const Point* pts[1] = { points };
	int npts[1] = { 4 };

	// ROI를 흰색으로 칠한다.
	Mat mask = Mat::zeros(edges.rows, edges.cols, CV_8UC1);
	fillPoly(mask, pts, npts, 1, Scalar(255));

	// 흰색으로 칠한 부분에 해당하는 edge만 리턴한다.
	Mat ROI_edges;
	bitwise_and(edges, mask, ROI_edges);

	return ROI_edges;
}


// 차선 색깔로 1차 필터링
void filterByColor(Mat image, Mat& filtered)
{
	// 흰색 차선(RGB)
	Scalar lower_white = Scalar(140, 140, 140);
	Scalar upper_white = Scalar(255, 255, 255);
	// 파란색 차선(HSV)
	Scalar lower_blue = Scalar(40, 0, 40);
	Scalar upper_blue = Scalar(130, 255, 255);

	Mat image_bgr = image.clone(), image_hsv;
	Mat white_mask, white_image;
	Mat blue_mask, blue_image;

	// 흰색 영역을 필터링해서 white_mask에 저장한다.
	inRange(image_bgr, lower_white, upper_white, white_mask);
	bitwise_and(image_bgr, image_bgr, white_image, white_mask);

	// 흰색 영역의 채도를 높인다.
	cvtColor(white_image, white_image, COLOR_BGR2HSV);
	vector<Mat> channel_white;
	split(white_image, channel_white);
	channel_white[1] *= 2;
	merge(channel_white, white_image);
	cvtColor(white_image, white_image, COLOR_HSV2BGR);

	// RGB 영상을 HSV 영상으로 변환한다.
	cvtColor(image_bgr, image_hsv, COLOR_BGR2HSV);

	// 파란색 영역을 필터링해서 blue_mask에 저장한다.
	inRange(image_hsv, lower_blue, upper_blue, blue_mask);
	bitwise_and(image_hsv, image_hsv, blue_image, blue_mask);

	// 파란색 영역의 채도를 높인다.
	vector<Mat> channel_blue;
	split(blue_image, channel_blue);
	channel_blue[1] *= 3;
	channel_blue[2] += 100;
	merge(channel_blue, blue_image);
	cvtColor(blue_image, blue_image, COLOR_HSV2BGR);

	// 흰색 영역과 파란색 영역을 합친다.
	addWeighted(white_image, 1.0, blue_image, 1.0, 0.0, filtered);
	imshow("filtered color region", filtered);
}


// 차선 검출
void detectLane(Mat& mark, Mat& detected, vector<Vec4i> lines)
{
	bool warning = false;
	vector<double> slopes;
	vector<Vec4i> filtered_lines;
	vector<Vec4i> warning_lines;

	// 각 선분의 기울기를 구한다
	int lines_size = (int)lines.size();
	for (int i = 0; i < lines_size; i++) {
		Vec4i line = lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];
		double slope;

		// 기울기를 계산한다.
		// zero division을 막기 위함
		if (x2 - x1 == 0)
			slope = 999.0;
		else
			slope = ((double)y2 - y1) / ((double)x2 - x1);

		// 기울기가 min_threshold 이상 max_threshold 이하인 line만 차선 후보로 간주한다.
		if (abs(slope) > slope_min_threshold && abs(slope) < slope_max_threshold) {
			slopes.push_back(slope);
			filtered_lines.push_back(line);
		}
		// 기울기가 1.6보다 크고 끝점의 x 좌표가 영상의 중앙에 위치하면 warning 상태이다.
		if (abs(slope) > 2 && (x1 > width * 0.4 && x1 < width * 0.5) && (x2 > width * 0.4 && x2 < width * 0.55)) {
			warning = true;
			warning_lines.push_back(line);
		}
	}
	// 기울기로 필터링한 후 어떤 선분이 남는지 확인하는 코드
	Mat tmp = frame.clone();
	for (Vec4i l : warning_lines) {
		line(tmp, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2);
	}
	imshow("warning_lines", tmp);


	// 왼쪽 line과 오른쪽 line을 고른다.
	// 기준 1: 오른쪽 차선의 기울기는 양수, 왼쪽 차선의 기울기는 음수
	// 기준 2: 오른쪽 차선의 x 좌표 평균값 > 중앙 
	vector<Vec4i> right_lines;
	vector<Vec4i> left_lines;
	int center_x = (int)(width * 0.5);  // 중앙 x 좌표

	int filtered_lines_size = (int)filtered_lines.size();
	for (int i = 0; i < filtered_lines_size; i++) {
		Vec4i line = filtered_lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];
		double slope = slopes[i];
		int avg_x = (x1 + x2) >> 1;  // 비트 연산자 사용해서 속도 ↑

		// 기울기 > 0, x 좌표 평균값 > 중앙 x 좌표이면 오른쪽 선으로 저장
		if (slope > 0 && avg_x > center_x)
			right_lines.push_back(line);
		// 기울기 < 0, x 좌표 평균값 < 중앙 x 좌표이면 왼쪽 선으로 저장
		else if (slope < 0 && avg_x < center_x)
			left_lines.push_back(line);
	}

	// ---------- 1) filtered line 평균 구하기

	// 오른쪽 filtered line 평균 구하기
	right_sum[0] = 0;
	right_sum[1] = 0;
	right_sum[2] = 0;
	right_sum[3] = 0;

	int right_lines_size = (int)right_lines.size();
	for (int i = 0; i < right_lines_size; i++) {
		Vec4i line = right_lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];

		right_sum[0] += x1;
		right_sum[1] += y1;
		right_sum[2] += x2;
		right_sum[3] += y2;
	}

	// 오른쪽 filtered line이 하나라도 있으면 평균 구하기
	if (right_lines_size != 0) {
		right_avg[0] = int((double)right_sum[0] / right_lines_size);
		right_avg[1] = int((double)right_sum[1] / right_lines_size);
		right_avg[2] = int((double)right_sum[2] / right_lines_size);
		right_avg[3] = int((double)right_sum[3] / right_lines_size);

		// 가장 옛날에 저장한 right line을 삭제하고 평균 line을 추가
		r_save_lines.erase(r_save_lines.begin(), r_save_lines.begin() + 1);
		r_save_lines.push_back(Vec4i(right_avg[0], right_avg[1], right_avg[2], right_avg[3]));

		r_save_cnt++;
	}

	// 왼쪽 filtered line 평균 구하기
	left_sum[0] = 0;
	left_sum[1] = 0;
	left_sum[2] = 0;
	left_sum[3] = 0;

	int left_lines_size = (int)left_lines.size();
	for (int i = 0; i < left_lines_size; i++) {
		Vec4i line = left_lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];

		left_sum[0] += x1;
		left_sum[1] += y1;
		left_sum[2] += x2;
		left_sum[3] += y2;
	}

	// 왼쪽 filtered line이 하나라도 있으면 좌표값 평균 구하기
	if (left_lines_size != 0) {
		left_avg[0] = int((double)left_sum[0] / left_lines_size);
		left_avg[1] = int((double)left_sum[1] / left_lines_size);
		left_avg[2] = int((double)left_sum[2] / left_lines_size);
		left_avg[3] = int((double)left_sum[3] / left_lines_size);

		// 가장 옛날에 저장한 left line을 삭제하고 평균 line을 추가
		l_save_lines.erase(l_save_lines.begin(), l_save_lines.begin() + 1);
		l_save_lines.push_back(Vec4i(left_avg[0], left_avg[1], left_avg[2], left_avg[3]));

		l_save_cnt++;
	}

	// ---------- 2) 최근 누적 line 10개의 평균 구하기

	// 오른쪽 누적 line 10개의 평균 구하기
	if (r_save_cnt > 10) {
		right_sum[0] = 0;
		right_sum[1] = 0;
		right_sum[2] = 0;
		right_sum[3] = 0;

		for (int i = 0; i < 10; i++) {
			Vec4i line = r_save_lines[i];
			right_sum[0] += line[0];
			right_sum[1] += line[1];
			right_sum[2] += line[2];
			right_sum[3] += line[3];
		}

		right_avg[0] = (int)((double)right_sum[0] / 10);
		right_avg[1] = (int)((double)right_sum[1] / 10);
		right_avg[2] = (int)((double)right_sum[2] / 10);
		right_avg[3] = (int)((double)right_sum[3] / 10);
	}

	// 왼쪽 누적 line 10개의 평균 구하기
	if (l_save_cnt > 10) {
		left_sum[0] = 0;
		left_sum[1] = 0;
		left_sum[2] = 0;
		left_sum[3] = 0;

		for (int i = 0; i < 10; i++) {
			Vec4i line = l_save_lines[i];
			left_sum[0] += line[0];
			left_sum[1] += line[1];
			left_sum[2] += line[2];
			left_sum[3] += line[3];
		}

		left_avg[0] = (int)((double)left_sum[0] / 10);
		left_avg[1] = (int)((double)left_sum[1] / 10);
		left_avg[2] = (int)((double)left_sum[2] / 10);
		left_avg[3] = (int)((double)left_sum[3] / 10);
	}

	// ---------- 3) 대표 line 구하기

	// 오른쪽 대표 line 좌표와 기울기
	int right_x = right_avg[0];
	int right_y = right_avg[1];
	double right_dx = (double)right_avg[0] - right_avg[2];
	double right_dy = (double)right_avg[1] - right_avg[3];
	// zero divison을 막기 위함
	if (right_dx == 0.0) right_dx = 0.001;
	if (right_dy == 0.0) right_dy = 0.001;
	double right_slope = right_dy / right_dx;

	// 왼쪽 대표 line 좌표와 기울기
	int left_x = left_avg[0];
	int left_y = left_avg[1];
	double left_dx = (double)left_avg[0] - left_avg[2];
	double left_dy = (double)left_avg[1] - left_avg[3];
	// zero divison을 막기 위함
	if (left_dx == 0.0) left_dx = 0.001;
	if (left_dy == 0.0) left_dy = 0.001;
	double left_slope = left_dy / left_dx;

	// 오른쪽 왼쪽 대표 line 좌표 구하기
	// y = mx + b → x = (y - b) / m
	// y = m(x - x0) + y0 = mx - mx0 + y0 → x = (y - y0) / m + x0
	int y1 = height;
	int y2 = (int)(height * 0.6);
	int right_x1 = (int)(((double)y1 - right_y) / right_slope + right_x);
	int right_x2 = (int)(((double)y2 - right_y) / right_slope + right_x);
	int left_x1 = (int)(((double)y1 - left_y) / left_slope + left_x);
	int left_x2 = (int)(((double)y2 - left_y) / left_slope + left_x);

	// 오른쪽 왼쪽 대표 line을 출력한다.
	line(mark, Point(right_x1, y1), Point(right_x2, y2), Scalar(255, 255, 255), 10);
	line(mark, Point(left_x1, y1), Point(left_x2, y2), Scalar(255, 255, 255), 10);

	// 검출한 차선을 칠한다.
	Point points[4];
	points[0] = Point(left_x1, y1);
	points[1] = Point(left_x2, y2);
	points[2] = Point(right_x2, y2);
	points[3] = Point(right_x1, y1);
	const Point* pts[1] = { points };
	int npts[] = { 4 };
	// warning 상태일 경우 차선을 빨간색으로 표시
	if (warning) {
		putText(mark, "WARNING", Point(50, 50), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 3);
		fillPoly(detected, pts, npts, 1, Scalar(0, 0, 255));
	}
	// safe 상태일 경우 차선을 초록색으로 표시
	else {
		putText(mark, "SAFE", Point(50, 50), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0), 3);
		fillPoly(detected, pts, npts, 1, Scalar(0, 255, 0));
	}
}


int main(int argc, char** argv) {

	// 동영상을 연다.
	VideoCapture video("clip4.mp4");

	// 동영상이 유효한지 확인한다.
	if (!video.isOpened()) {
		cout << "동영상 파일을 열거나 찾을 수 없습니다." << endl;
		return -1;
	}
	
	// 누적 line 벡터를 초기화한다.
	for (int i = 0; i < 10; i++) {
		r_save_lines.push_back(Vec4i(0, 0, 0, 0));
		l_save_lines.push_back(Vec4i(0, 0, 0, 0));
	}

	// 동영상으로부터 영상을 읽어 frame에 넣는다.
	video.read(frame);

	// 영상이 유효한지 확인한다.
	if (frame.empty()) {
		destroyAllWindows();
		return -1;
	}

	// 동영상 파일의 초당 프레임 수를 불러온다.
	double fps = video.get(CAP_PROP_FPS);

	// 프레임 사이의 간격을 계산한다.
	int delay = cvRound(1000 / fps);

	// 영상의 너비와 높이를 계산한다.
	width = frame.cols;   // 1920
	height = frame.rows;  // 1080

	// 동영상이 끝날 때까지 영상을 출력한다.
	do {
		// 동영상으로부터 영상을 읽어 frame에 넣는다.
		video.read(frame);

		// 영상이 유효한지 확인한다.
		if (frame.empty()) {
			destroyAllWindows();
			break;
		}

		// 컬러 범위를 통해 후보 영역을 필터링한다.
		Mat ROI_colors;
		filterByColor(frame, ROI_colors);

		// RGB 영상을 그레이스케일 영상으로 변환한다.
		cvtColor(ROI_colors, gray, COLOR_BGR2GRAY);

		// 캐니 에지를 검출한다.
		GaussianBlur(gray, gray, Size(3, 3), 0, 0);
		Canny(gray, edges, 100, 200);

		// 차선을 검출할 영역을 제한한다.
		Point points[4];
		points[0] = Point((int)(width * 0.15), (int)(height * 0.8));
		points[1] = Point((int)(width * 0.25), (int)(height * 0.55));
		points[2] = Point((int)(width * 0.55), (int)(height * 0.55));
		points[3] = Point((int)(width * 0.7), (int)(height * 0.8));
		edges = setROI(edges, points);

		//namedWindow("Image Edges2");
		imshow("ROI Canny Edges", edges);

		//Mat uImage_edges;
		//edges.copyTo(uImage_edges);

		// line을 검출한다.
		vector<Vec4i> lines;
		HoughLinesP(edges, lines, rho, theta, hough_threshold, minLineLength, maxLineGap);

		// 차선을 검출한다.
		Mat mark = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		Mat detected = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		detectLane(mark, detected, lines);
		imshow("mark", mark);
		imshow("detected", detected);
		imshow("frame", frame);

		// 원본 영상에 차선을 표시한다.
		mark.copyTo(frame, mark);
		addWeighted(frame, 1, detected, 0.2, 0.0, result);

		// 결과 영상을 출력한다.
		imshow("final", result);

		// ESC 키를 누르면 종료한다.
		// delay 값을 통해 원본 영상과 같은 프레임 간격으로 영상을 출력한다.
		if (waitKey(delay) == 27) break;
	} while (!frame.empty());

	return 0;
}
