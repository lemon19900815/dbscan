// testglog.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <future>
#include <functional>
#include <vector>
#include <cstdint>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>

#include "glog_helper.h"
#include "dbscan.h"
#include <iostream>

int main()
{
  float eps = 4;//0.75 * 0.75;
  int32_t min_pts = 4;

  DBSCAN db(eps, min_pts);

  std::vector<Point> points = {
    {0,1},
    {1,0},
    {1,1},
    {0,0},
    {2,2},
    {2,3},
    {3,3},
    {3,8},
    {10,9},
    {12,10},
    {10,13},
    {11,12}
  };

  std::vector<int32_t> cluster;
  if (!db.Fit(points, cluster)) {
    std::cout << "DBSCAN cluster fit failed." << std::endl;
    return -1;
  }

  std::string sep = "";
  std::stringstream ss;
  for (auto& id : cluster) {
    ss << sep << id;
    sep = ", ";
  }
  std::cout << "cluster: " << ss.str() << std::endl;

  //system("pause");
  return 0;
}

