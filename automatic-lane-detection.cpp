#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;


// ������ ����
Mat frame, gray, edges, result;
int width, height;


// HoughLinesP �Լ� ����
double rho = 2;
double theta = CV_PI / 180;
int hough_threshold = 100;
double minLineLength = 100;
double maxLineGap = 250;


// ���� threshold
double slope_min_threshold = 0.65;
double slope_max_threshold = 1.2;


// ���� line ���� ���� ����
vector<Vec4i> r_save_lines;
int right_sum[4] = { 0 };
int right_avg[4];
int r_save_cnt = 0;
vector<Vec4i> l_save_lines;
int left_sum[4] = { 0 };
int left_avg[4];
int l_save_cnt = 0;


// ȭ�鿡�� line�� ������ ����(ROI) ����
Mat setROI(Mat edges, Point* points)
{
	const Point* pts[1] = { points };
	int npts[1] = { 4 };

	// ROI�� ������� ĥ�Ѵ�.
	Mat mask = Mat::zeros(edges.rows, edges.cols, CV_8UC1);
	fillPoly(mask, pts, npts, 1, Scalar(255));

	// ������� ĥ�� �κп� �ش��ϴ� edge�� �����Ѵ�.
	Mat ROI_edges;
	bitwise_and(edges, mask, ROI_edges);

	return ROI_edges;
}


// ���� ����� 1�� ���͸�
void filterByColor(Mat image, Mat& filtered)
{
	// ��� ����(RGB)
	Scalar lower_white = Scalar(140, 140, 140);
	Scalar upper_white = Scalar(255, 255, 255);
	// �Ķ��� ����(HSV)
	Scalar lower_blue = Scalar(40, 0, 40);
	Scalar upper_blue = Scalar(130, 255, 255);

	Mat image_bgr = image.clone(), image_hsv;
	Mat white_mask, white_image;
	Mat blue_mask, blue_image;

	// ��� ������ ���͸��ؼ� white_mask�� �����Ѵ�.
	inRange(image_bgr, lower_white, upper_white, white_mask);
	bitwise_and(image_bgr, image_bgr, white_image, white_mask);

	// ��� ������ ä���� ���δ�.
	cvtColor(white_image, white_image, COLOR_BGR2HSV);
	vector<Mat> channel_white;
	split(white_image, channel_white);
	channel_white[1] *= 2;
	merge(channel_white, white_image);
	cvtColor(white_image, white_image, COLOR_HSV2BGR);

	// RGB ������ HSV �������� ��ȯ�Ѵ�.
	cvtColor(image_bgr, image_hsv, COLOR_BGR2HSV);

	// �Ķ��� ������ ���͸��ؼ� blue_mask�� �����Ѵ�.
	inRange(image_hsv, lower_blue, upper_blue, blue_mask);
	bitwise_and(image_hsv, image_hsv, blue_image, blue_mask);

	// �Ķ��� ������ ä���� ���δ�.
	vector<Mat> channel_blue;
	split(blue_image, channel_blue);
	channel_blue[1] *= 3;
	channel_blue[2] += 100;
	merge(channel_blue, blue_image);
	cvtColor(blue_image, blue_image, COLOR_HSV2BGR);

	// ��� ������ �Ķ��� ������ ��ģ��.
	addWeighted(white_image, 1.0, blue_image, 1.0, 0.0, filtered);
	imshow("filtered color region", filtered);
}


// ���� ����
void detectLane(Mat& mark, Mat& detected, vector<Vec4i> lines)
{
	bool warning = false;
	vector<double> slopes;
	vector<Vec4i> filtered_lines;
	vector<Vec4i> warning_lines;

	// �� ������ ���⸦ ���Ѵ�
	int lines_size = (int)lines.size();
	for (int i = 0; i < lines_size; i++) {
		Vec4i line = lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];
		double slope;

		// ���⸦ ����Ѵ�.
		// zero division�� ���� ����
		if (x2 - x1 == 0)
			slope = 999.0;
		else
			slope = ((double)y2 - y1) / ((double)x2 - x1);

		// ���Ⱑ min_threshold �̻� max_threshold ������ line�� ���� �ĺ��� �����Ѵ�.
		if (abs(slope) > slope_min_threshold && abs(slope) < slope_max_threshold) {
			slopes.push_back(slope);
			filtered_lines.push_back(line);
		}
		// ���Ⱑ 1.6���� ũ�� ������ x ��ǥ�� ������ �߾ӿ� ��ġ�ϸ� warning �����̴�.
		if (abs(slope) > 2 && (x1 > width * 0.4 && x1 < width * 0.5) && (x2 > width * 0.4 && x2 < width * 0.55)) {
			warning = true;
			warning_lines.push_back(line);
		}
	}
	// ����� ���͸��� �� � ������ ������ Ȯ���ϴ� �ڵ�
	Mat tmp = frame.clone();
	for (Vec4i l : warning_lines) {
		line(tmp, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2);
	}
	imshow("warning_lines", tmp);


	// ���� line�� ������ line�� ����.
	// ���� 1: ������ ������ ����� ���, ���� ������ ����� ����
	// ���� 2: ������ ������ x ��ǥ ��հ� > �߾� 
	vector<Vec4i> right_lines;
	vector<Vec4i> left_lines;
	int center_x = (int)(width * 0.5);  // �߾� x ��ǥ

	int filtered_lines_size = (int)filtered_lines.size();
	for (int i = 0; i < filtered_lines_size; i++) {
		Vec4i line = filtered_lines[i];
		int x1 = line[0];
		int y1 = line[1];
		int x2 = line[2];
		int y2 = line[3];
		double slope = slopes[i];
		int avg_x = (x1 + x2) >> 1;  // ��Ʈ ������ ����ؼ� �ӵ� ��

		// ���� > 0, x ��ǥ ��հ� > �߾� x ��ǥ�̸� ������ ������ ����
		if (slope > 0 && avg_x > center_x)
			right_lines.push_back(line);
		// ���� < 0, x ��ǥ ��հ� < �߾� x ��ǥ�̸� ���� ������ ����
		else if (slope < 0 && avg_x < center_x)
			left_lines.push_back(line);
	}

	// ---------- 1) filtered line ��� ���ϱ�

	// ������ filtered line ��� ���ϱ�
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

	// ������ filtered line�� �ϳ��� ������ ��� ���ϱ�
	if (right_lines_size != 0) {
		right_avg[0] = int((double)right_sum[0] / right_lines_size);
		right_avg[1] = int((double)right_sum[1] / right_lines_size);
		right_avg[2] = int((double)right_sum[2] / right_lines_size);
		right_avg[3] = int((double)right_sum[3] / right_lines_size);

		// ���� ������ ������ right line�� �����ϰ� ��� line�� �߰�
		r_save_lines.erase(r_save_lines.begin(), r_save_lines.begin() + 1);
		r_save_lines.push_back(Vec4i(right_avg[0], right_avg[1], right_avg[2], right_avg[3]));

		r_save_cnt++;
	}

	// ���� filtered line ��� ���ϱ�
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

	// ���� filtered line�� �ϳ��� ������ ��ǥ�� ��� ���ϱ�
	if (left_lines_size != 0) {
		left_avg[0] = int((double)left_sum[0] / left_lines_size);
		left_avg[1] = int((double)left_sum[1] / left_lines_size);
		left_avg[2] = int((double)left_sum[2] / left_lines_size);
		left_avg[3] = int((double)left_sum[3] / left_lines_size);

		// ���� ������ ������ left line�� �����ϰ� ��� line�� �߰�
		l_save_lines.erase(l_save_lines.begin(), l_save_lines.begin() + 1);
		l_save_lines.push_back(Vec4i(left_avg[0], left_avg[1], left_avg[2], left_avg[3]));

		l_save_cnt++;
	}

	// ---------- 2) �ֱ� ���� line 10���� ��� ���ϱ�

	// ������ ���� line 10���� ��� ���ϱ�
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

	// ���� ���� line 10���� ��� ���ϱ�
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

	// ---------- 3) ��ǥ line ���ϱ�

	// ������ ��ǥ line ��ǥ�� ����
	int right_x = right_avg[0];
	int right_y = right_avg[1];
	double right_dx = (double)right_avg[0] - right_avg[2];
	double right_dy = (double)right_avg[1] - right_avg[3];
	// zero divison�� ���� ����
	if (right_dx == 0.0) right_dx = 0.001;
	if (right_dy == 0.0) right_dy = 0.001;
	double right_slope = right_dy / right_dx;

	// ���� ��ǥ line ��ǥ�� ����
	int left_x = left_avg[0];
	int left_y = left_avg[1];
	double left_dx = (double)left_avg[0] - left_avg[2];
	double left_dy = (double)left_avg[1] - left_avg[3];
	// zero divison�� ���� ����
	if (left_dx == 0.0) left_dx = 0.001;
	if (left_dy == 0.0) left_dy = 0.001;
	double left_slope = left_dy / left_dx;

	// ������ ���� ��ǥ line ��ǥ ���ϱ�
	// y = mx + b �� x = (y - b) / m
	// y = m(x - x0) + y0 = mx - mx0 + y0 �� x = (y - y0) / m + x0
	int y1 = height;
	int y2 = (int)(height * 0.6);
	int right_x1 = (int)(((double)y1 - right_y) / right_slope + right_x);
	int right_x2 = (int)(((double)y2 - right_y) / right_slope + right_x);
	int left_x1 = (int)(((double)y1 - left_y) / left_slope + left_x);
	int left_x2 = (int)(((double)y2 - left_y) / left_slope + left_x);

	// ������ ���� ��ǥ line�� ����Ѵ�.
	line(mark, Point(right_x1, y1), Point(right_x2, y2), Scalar(255, 255, 255), 10);
	line(mark, Point(left_x1, y1), Point(left_x2, y2), Scalar(255, 255, 255), 10);

	// ������ ������ ĥ�Ѵ�.
	Point points[4];
	points[0] = Point(left_x1, y1);
	points[1] = Point(left_x2, y2);
	points[2] = Point(right_x2, y2);
	points[3] = Point(right_x1, y1);
	const Point* pts[1] = { points };
	int npts[] = { 4 };
	// warning ������ ��� ������ ���������� ǥ��
	if (warning) {
		putText(mark, "WARNING", Point(50, 50), FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255), 3);
		fillPoly(detected, pts, npts, 1, Scalar(0, 0, 255));
	}
	// safe ������ ��� ������ �ʷϻ����� ǥ��
	else {
		putText(mark, "SAFE", Point(50, 50), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 0), 3);
		fillPoly(detected, pts, npts, 1, Scalar(0, 255, 0));
	}
}


int main(int argc, char** argv) {

	// �������� ����.
	VideoCapture video("clip4.mp4");

	// �������� ��ȿ���� Ȯ���Ѵ�.
	if (!video.isOpened()) {
		cout << "������ ������ ���ų� ã�� �� �����ϴ�." << endl;
		return -1;
	}
	
	// ���� line ���͸� �ʱ�ȭ�Ѵ�.
	for (int i = 0; i < 10; i++) {
		r_save_lines.push_back(Vec4i(0, 0, 0, 0));
		l_save_lines.push_back(Vec4i(0, 0, 0, 0));
	}

	// ���������κ��� ������ �о� frame�� �ִ´�.
	video.read(frame);

	// ������ ��ȿ���� Ȯ���Ѵ�.
	if (frame.empty()) {
		destroyAllWindows();
		return -1;
	}

	// ������ ������ �ʴ� ������ ���� �ҷ��´�.
	double fps = video.get(CAP_PROP_FPS);

	// ������ ������ ������ ����Ѵ�.
	int delay = cvRound(1000 / fps);

	// ������ �ʺ�� ���̸� ����Ѵ�.
	width = frame.cols;   // 1920
	height = frame.rows;  // 1080

	// �������� ���� ������ ������ ����Ѵ�.
	do {
		// ���������κ��� ������ �о� frame�� �ִ´�.
		video.read(frame);

		// ������ ��ȿ���� Ȯ���Ѵ�.
		if (frame.empty()) {
			destroyAllWindows();
			break;
		}

		// �÷� ������ ���� �ĺ� ������ ���͸��Ѵ�.
		Mat ROI_colors;
		filterByColor(frame, ROI_colors);

		// RGB ������ �׷��̽����� �������� ��ȯ�Ѵ�.
		cvtColor(ROI_colors, gray, COLOR_BGR2GRAY);

		// ĳ�� ������ �����Ѵ�.
		GaussianBlur(gray, gray, Size(3, 3), 0, 0);
		Canny(gray, edges, 100, 200);

		// ������ ������ ������ �����Ѵ�.
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

		// line�� �����Ѵ�.
		vector<Vec4i> lines;
		HoughLinesP(edges, lines, rho, theta, hough_threshold, minLineLength, maxLineGap);

		// ������ �����Ѵ�.
		Mat mark = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		Mat detected = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
		detectLane(mark, detected, lines);
		imshow("mark", mark);
		imshow("detected", detected);
		imshow("frame", frame);

		// ���� ���� ������ ǥ���Ѵ�.
		mark.copyTo(frame, mark);
		addWeighted(frame, 1, detected, 0.2, 0.0, result);

		// ��� ������ ����Ѵ�.
		imshow("final", result);

		// ESC Ű�� ������ �����Ѵ�.
		// delay ���� ���� ���� ����� ���� ������ �������� ������ ����Ѵ�.
		if (waitKey(delay) == 27) break;
	} while (!frame.empty());

	return 0;
}
