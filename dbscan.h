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

  // �㷨ʵ�ֲο�����־��-����ѧϰ������DBSCAN��α��������
  // �ӿ�����sklearn.cluster.DBSCAN
  bool Fit(std::vector<Point>& points,
    std::vector<int32_t>& cluster_ids);

private:
  // ��2�����ϵĲ
  std::vector<int32_t> DifferenceSet(
    std::vector<int32_t>& src, std::vector<int32_t>& dels);

  // ��2�����ϵĲ���
  std::vector<int32_t> UnionSet(
    std::vector<int32_t>& a, std::vector<int32_t>& b);

  // ��2�����ϵĽ���
  std::vector<int32_t> Intersection(
    std::vector<int32_t>& a, std::vector<int32_t>& b);

  // �������Ķ��󼯺�
  std::vector<int32_t> BuildCoreObjects(std::vector<Point>& points);

  // ��ȡpoints[pos]�Ľ��ڼ���
  std::vector<int32_t> GetNeighborhoods(
    int32_t pos, std::vector<Point>& points);

  // ��ȡpoints[pos]�Ľ�������
  int32_t GetNeighborhoodsNum(int32_t pos, std::vector<Point>& points);

  // ŷ�Ͼ���
  float Distance(Point& p1, Point& p2);

  // ��ѯ2��֮��ľ���
  float GetDistance(int32_t pos1, int32_t pos2);

  // �������е�֮�����������
  void CalcDistance(std::vector<Point>& points);

private:
  float eps_;
  int32_t min_pts_;
  std::vector<std::vector<float>> matrix_dist_;
};
