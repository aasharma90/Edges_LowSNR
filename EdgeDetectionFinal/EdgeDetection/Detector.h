#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <mutex>
#include "tools.h"

using namespace std;
using namespace cv;

#define TYPE CV_64F
#define BOOL CV_8U

#define ZERO double(0)
#define MINUS double(-1)
#define FALSE Scalar(0)
#define CIRCLE int(360)
#define PI double(3.141592653589793238463)

typedef unsigned int uint;

typedef struct MyParam{

	/* Remove bottom level lines with response lower than removeEpsilon x sigma */
	double removeEpsilon = 0.248;

	/* maxTurn is the maximal turn of the bottom levels, and 2 x maxTurn is the maximal turn of the topmost level */
	double maxTurn = 35;

	/* Edges with numFact overlap cannot be joined together to the final edge map */
	double nmsFact = 0.75;

	/* K splitPoints to consider in every interface */
	int splitPoints = 0; // 0 = allPoints

	/* Edges with length greater than minContrast should pass the min-constrast test */
	uint minContrast = 9; // 0 = noTest

	/* Norm Type */
	uint normType = 0; // 0 = max(dx,dy), 1 = dx+dy, 2  = sqrt(dx^2+dy^2);

	uint maxNumOfEdges = 50;
	double sigma = 0.1;
	uint patchSize = 5;
    uint filterWidth = 2;
	bool parallel = true;
	uint parallelJump = 1;
	bool noisyImage = true; // false = lowerThreshold
	uint minLevelOfStitching = 0; // 0 = stichAll
	uint slidingWindow = (uint)pow(2,8)+1; // 0 = allImage
	bool printToScreen = true;
	bool interpolation = true;
	bool fibers = false;
	bool generate_Laplacian = false; // To generate Laplacian matting affinity matrix or not
	uint generate_Laplacian_win_size = 4; // Default win_size = 4 (9x9 neighbourhood)

	/* Returns the maximum of two matrices */
	void matrixMaximum(const Mat& x, const Mat& y, const Mat& dst){
		assert(x.size() == y.size());
		Mat m1 = Mat(x.rows, x.cols, TYPE, ZERO);
		Mat m2 = Mat(y.rows, y.cols, TYPE, ZERO);
		cv::max(x, m1, m1);
		cv::max(y, m2, m2);
		Mat m3 = (m1+m2);
		m3.copyTo(dst);
	}
} MyParam;

typedef struct Handle{
	uint m;
	uint n;
	uint N;
	Mat S;
	Size rSize;
	uint R = 0;
	uint L = 1;
	uint C = 2;
	uint SC = 3;
	uint minC = 4;
	uint maxC = 5;
	uint I0S0 = 6;
	uint S0I1 = 7;
	uint A0 = 8;
	uint A1 = 9;
	uint TOTAL = 10;
} Handle;

class Detector {
	private:
		Mat _I;
		Mat _dX;
		Mat _dY;
		Mat _dX_intermediate;
		Mat _dY_intermediate;
		Mat _E;
		uint _w;
		Mat _filter;
		Mat _filter_intermediate;
		MyParam _prm;
		Handle _handle;
		unordered_map<uint, Mat>* _data;
		Mat* _pixelScores;
		unordered_map<uint, Mat> _pixels;
		bool _debug = false;
		std::mutex _mtx;
		uint _maxLevel;
		Mat W_Laplacian;
		Mat A_Laplacian;

		/* Bottom Level Processing */
		void getBottomLevelSimple(Mat& S, uint index);
		int getLine(int x0, int y0, int x1, int y1, Mat& P, Mat& F);
		void getLineFilter(int x0, int y0, int x1, int y1, Mat& F);
		void interpLine(double x, double y, Mat& F, double value);
		void getVerticesFromPatchIndices(uint e, uint  v, uint m, uint n, uint& x, uint& y);

		/* Core Functions*/
		void getBestSplittingPoints(Mat& split, Mat& dst, uint index);
		void mergeTilesSimple(const Mat& S1, const Mat& S2, uint index, uint level, bool verticalSplit);
		void findBestResponse(Mat& edge1, Mat& split, Mat& edge2, uint index, uint level);

		/* Sub Functions */
		void subIm(const Mat& Ssrc, uint x0, uint y0, uint x1, uint y1, Mat& Sdst);
		void getEdgeIndices(const Mat& S, vector<Mat>& v);
		void addIndices(Mat& table, Mat& ind0, Mat& s0, Mat& ind1);
		bool angleInRange(uint ind0, uint s0, uint ind1, uint level);
		bool angleInRange(double ang0, double ang1, uint level);
		double getThreshold(double len){
			double fact = _prm.noisyImage ? 1 : 0.0;
			return _prm.sigma*(fact*0.14+sqrt(2 * log(6 * _handle.N) / _w / len / 2));
		};
		uint getSideLength(uint m, uint n, uint e);
		bool insertValueToMap(uint index, const double& key, const Mat& value);
		int getAngle(double dx, double dy);
		void lock();
		void unlock();

		/* Post Processing */
		void getScores();
		void removeKey(unordered_map<uint, Mat>& data, uint key);
		bool addEdge(unordered_map<uint, Mat>& data, uint curKey, Mat& E, uint level);

	public:
		Detector(const Mat& I, const MyParam& prm);
		~Detector();
		Mat runIm();
		Mat getE(){ return _E;};
		Mat getPixelScores(){
			Mat P = _pixelScores[0];
			P = P / maxValue(P);
			return _pixelScores[0];
		};
		void beamCurves(uint index, uint level, Mat* S);
		int _my_experiment = 0;
};
