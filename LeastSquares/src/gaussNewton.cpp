#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {

  // initialization
  double ref_a = 1.0, ref_b = 2.0, ref_c = 1.0; 
  double est_a = 2.0, est_b = -1.0, est_c = 5.0;
  int N = 100;                         
  double w_sigma = 1.0;                
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng;                         

  // data
  vector<double> x_data, y_data;     
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ref_a * x * x + ref_b * x + ref_c) + rng.gaussian(w_sigma * w_sigma));
  }

  int iterations = 100;  
  double cost = 0, lastCost = 0; 

  // least square
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for (int iter = 0; iter < iterations; iter++) {

    Matrix3d H = Matrix3d::Zero(); 
    Vector3d b = Vector3d::Zero(); 
    cost = 0;

    for (int i = 0; i < N; i++) {
      double xi = x_data[i], yi = y_data[i]; 
      double error = yi - exp(est_a * xi * xi + est_b * xi + est_c);
      Vector3d J; 
      J[0] = -xi * xi * exp(est_a * xi * xi + est_b * xi + est_c);
      J[1] = -xi * exp(est_a * xi * xi + est_b * xi + est_c); 
      J[2] = -exp(est_a * xi * xi + est_b * xi + est_c); 

      H += inv_sigma * inv_sigma * J * J.transpose();
      b += -inv_sigma * inv_sigma * error * J;

      cost += error * error;
    }

    Vector3d dx = H.ldlt().solve(b);
    if (isnan(dx[0])) {
      cout << "result is nan!" << endl;
      break;
    }

    if (iter > 0 && cost >= lastCost) {
      cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
      break;
    }

    est_a += dx[0];
    est_b += dx[1];
    est_c += dx[2];

    lastCost = cost;

    cout << "total cost: " << cost << ", \t\t update: " << dx.transpose() <<
         "\t\testimated params: " << est_a << "," << est_b << "," << est_c << endl;
  }

  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  cout << "estimated abc = " << est_a << ", " << est_b << ", " << est_c << endl;
  return 0;
}
