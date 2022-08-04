#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>

struct Point {
  float x_;
  float y_;
};

class DBSCAN {
public:
  DBSCAN(float eps, int32_t min_pts);
  ~DBSCAN();

  // 算法实现参考《周志华-机器学习》关于DBSCAN的伪代码描述
  // 接口类似sklearn.cluster.DBSCAN
  bool Fit(std::vector<Point>& points,
    std::vector<int32_t>& cluster_ids);

private:
  // 求2个集合的差集
  std::vector<int32_t> DifferenceSet(
    std::vector<int32_t>& src, std::vector<int32_t>& dels);

  // 求2个集合的并集
  std::vector<int32_t> UnionSet(
    std::vector<int32_t>& a, std::vector<int32_t>& b);

  // 求2个集合的交集
  std::vector<int32_t> Intersection(
    std::vector<int32_t>& a, std::vector<int32_t>& b);

  // 构建核心对象集合
  std::vector<int32_t> BuildCoreObjects(std::vector<Point>& points);

  // 获取points[pos]的近邻集合
  std::vector<int32_t> GetNeighborhoods(
    int32_t pos, std::vector<Point>& points);

  // 获取points[pos]的近邻数量
  int32_t GetNeighborhoodsNum(int32_t pos, std::vector<Point>& points);

  // 欧氏距离
  float Distance(Point& p1, Point& p2);

  // 查询2点之间的距离
  float GetDistance(int32_t pos1, int32_t pos2);

  // 计算所有点之间的两两距离
  void CalcDistance(std::vector<Point>& points);

private:
  float eps_;
  int32_t min_pts_;
  std::vector<std::vector<float>> matrix_dist_;
};
