#include "dbscan.h"

DBSCAN::DBSCAN(float eps, int32_t min_pts) {
  eps_ = eps;
  min_pts_ = min_pts;
}

DBSCAN::~DBSCAN() {
}

// �㷨ʵ�ֲο�����־��-����ѧϰ������DBSCAN��α��������
bool DBSCAN::Fit(std::vector<Point>& points,
    std::vector<int32_t>& cluster_ids) {

  if (points.size() < min_pts_)
    return false;

  // �������е�֮�����������
  CalcDistance(points);

  auto omega = BuildCoreObjects(points);
  if (omega.empty())
    return false;

  std::vector<int32_t> un_readsets; // δ���ʵ���������
  for (int32_t i = 0; i < points.size(); ++i) {
    un_readsets.push_back(i);
  }

  int32_t num_cluster = 0;
  std::vector<std::vector<int32_t>> res;

  std::random_shuffle(omega.begin(), omega.end());

  while (!omega.empty()) {
    auto old_un_readsets = un_readsets;

    // ���ѡȡһ�����Ķ���
    auto index = omega.back();
    omega.pop_back();

    // ��ʼ������
    std::vector<int32_t> q = { index };

    // F\{index}
    un_readsets = DifferenceSet(un_readsets, q);

    while (!q.empty()) {
      index = q.front();

      if (q.size() > 1)
        std::swap(q.back(), q.front());
      q.pop_back();

      auto neighborhoods = GetNeighborhoods(index, points);
      if (neighborhoods.size() >= min_pts_) {
        auto delta = Intersection(neighborhoods, un_readsets);
        if(delta.empty())
          continue;

        q = UnionSet(q, delta);
        un_readsets = DifferenceSet(un_readsets, delta);
      }
    }

    num_cluster += 1;

    auto ck = DifferenceSet(old_un_readsets, un_readsets);
    res.push_back(ck);

    omega = DifferenceSet(omega, ck);
  }

  // �����������Ϊ����
  cluster_ids.resize(points.size(), -1);

  int32_t cluster_id = 1;
  for (auto& cluster : res) {
    if (cluster.size() >= min_pts_) {
      for (auto& idx : cluster) {
        cluster_ids[idx] = cluster_id;
      }
      cluster_id += 1;
    }
  }

  return true;
}

// ��2�����ϵĲ
std::vector<int32_t> DBSCAN::DifferenceSet(
    std::vector<int32_t>& src, std::vector<int32_t>& dels) {

  std::unordered_map<int32_t, bool> hash;
  for (auto& idx : src) {
    hash[idx] = true;
  }

  for (auto& idx : dels) {
    hash.erase(idx);
  }

  std::vector<int32_t> res;
  for (auto& it : hash) {
    res.push_back(it.first);
  }

  return res;
}

// ��2�����ϵĲ���
std::vector<int32_t> DBSCAN::UnionSet(
    std::vector<int32_t>& a, std::vector<int32_t>& b) {

  std::vector<int32_t> res;
  std::unordered_map<int32_t, bool> hash;
  for (auto& idx : a) {
    hash[idx] = true;
    res.push_back(idx);
  }

  for (auto& idx : b) {
    if (hash.find(idx) == hash.end())
      res.push_back(idx);
  }
  return res;
}

// ��2�����ϵĽ���
std::vector<int32_t> DBSCAN::Intersection(
    std::vector<int32_t>& a, std::vector<int32_t>& b) {

  std::unordered_map<int32_t, bool> hash;
  for (auto& idx : a) {
    hash[idx] = true;
  }

  std::vector<int32_t> res;
  for (auto& idx : b) {
    if (hash.find(idx) != hash.end())
      res.push_back(idx);
  }
  return res;
}

// �������Ķ��󼯺�
std::vector<int32_t> DBSCAN::BuildCoreObjects(std::vector<Point>& points) {
  std::vector<int32_t> omega;
  for (int32_t i = 0; i < points.size(); ++i) {
    if (GetNeighborhoodsNum(i, points) >= min_pts_)
      omega.push_back(i);
  }
  return omega;
}

// ��ȡpoints[pos]�Ľ��ڼ���
std::vector<int32_t> DBSCAN::GetNeighborhoods(
    int32_t pos, std::vector<Point>& points) {

  std::vector<int32_t> neighborhoods;
  for (int32_t i = 0; i < points.size(); ++i) {
    if (GetDistance(pos, i) <= eps_)
      neighborhoods.push_back(i);
  }
  return neighborhoods;
}

// ��ȡpoints[pos]�Ľ�������
int32_t DBSCAN::GetNeighborhoodsNum(int32_t pos, std::vector<Point>& points) {
  return GetNeighborhoods(pos, points).size();
}

// ŷ�Ͼ���
float DBSCAN::Distance(Point& p1, Point& p2) {
  return std::sqrt(std::pow(p1.x_-p2.x_, 2) + std::pow(p1.y_-p2.y_, 2));
}

// ��ѯ2��֮��ľ���
float DBSCAN::GetDistance(int32_t pos1, int32_t pos2) {
  return matrix_dist_[pos1][pos2];
}

// �������е�֮�����������
void DBSCAN::CalcDistance(std::vector<Point>& points) {
  auto num_pts = points.size();

  matrix_dist_.resize(num_pts);
  for (auto& info : matrix_dist_) {
    info.resize(num_pts);
  }

  float dist = 0.0f;
  for (int32_t i = 0; i < num_pts; ++i) {
    for (int32_t j = 0; j < num_pts; ++j) {
      dist = Distance(points[i], points[j]);
      matrix_dist_[i][j] = dist;
      matrix_dist_[j][i] = dist;
    }
  }
}

