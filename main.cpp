#include <iostream>
#include <list>
#include <unordered_map>
#include <string>
#include <vector>
#include <cmath>
#include <climits>
#include <algorithm>
#include <unordered_set>
#include <stack>
#include <set>
#include <queue>
#include <time.h>
#include <cstring>
#include <random>
#include <map>
#include "container/emplace_back_check.h"
#include "container/rvalue_check.h"

using namespace std;

struct TreeNode {
  int val;
  TreeNode *left;
  TreeNode *right;

  TreeNode() : val(0), left(nullptr), right(nullptr) {}
  TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
  TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};

//class Node {
//  int row;
//  int col;
//  Node(int row, int col) {
//    this->row = row;
//    this->col = col;
//  }
//};

class ParentTreeNode {
public:
  int val;
  ParentTreeNode *parent, *left, *right;
};

class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};

class LRUCache {
public:
	LRUCache(int capacity) : capacity(capacity) {}
	int get(int key) {
		if (pos.find(key) != pos.end()){
			put(key, pos[key]->second);
			return pos[key]->second;
		}
		return -1;
	}
	void put(int key, int value) {
		if (pos.find(key) != pos.end())
			recent.erase(pos[key]);
		else if (recent.size() >= capacity) {
			pos.erase(recent.back().first);
			recent.pop_back();
		}
		recent.push_front({ key, value });
		pos[key] = recent.begin();
	}

	void out() {
	  for(unordered_map<int, list<pair<int, int>>::iterator>::iterator iter = pos.begin(); iter != pos.end(); iter++) {
	    cout<<"key value is "<< iter->first << " the mapped value is " << endl;
	  }
	}

private:
	int capacity;
	list<pair<int, int>> recent;
	unordered_map<int, list<pair<int, int>>::iterator> pos;  //value存储的是一个迭代器
};

struct ListNode {
  int val;
  ListNode *next;

  ListNode(int x) : val(x), next(NULL) {}
};

class MinStack {
public:
    /** initialize your data structure here. */
    MinStack() {
      st.reserve(cap);
      min_st.resize(cap);
    }

    void push(int x) {
      point++;
      st[point] = x;
      if (point == 0 || x < min_st[point-1]) {
        min_st[point] = x;
      } else {
        min_st[point] = min_st[point-1];
      }
    }

    void pop() {
      point--;
      st.emplace_back();
    }

    int top() {
      return st[point];
    }

    int getMin() {
      return min_st[point];
    }

private:
    int point = -1;
    int cap = 100;
    vector<int> st;
    vector<int> min_st;
    std::stack<int> sta{};
};

//https://leetcode.com/problems/binary-search-tree-iterator/
class BSTIterator {
public:
    BSTIterator(TreeNode* root) {
      iteration(root);
    }

    /** @return the next smallest number */
    int next() {
      TreeNode* cur = st.top();
      st.pop();
      iteration(cur->right);
      return cur->val;
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
      return !st.empty();
    }

    void iteration(TreeNode* root) {
      while (nullptr != root) {
        st.emplace(root);
        root = root->left;
      }
    }

private:
  std::stack<TreeNode*> st{};
};

//https://leetcode.com/problems/kth-largest-element-in-a-stream/
class KthLargest {
public:
    KthLargest(int k, vector<int>& nums) {
      priority = k;
      for (auto& num : nums) {
        add(num);
      }
    }

    int add(int val) {
      if (_pq.size() == priority && _pq.top() >= val) {
        return _pq.top();
      } else if (_pq.size() < priority) {
        _pq.emplace(val);
      } else {
        _pq.pop();
        _pq.emplace(val);
      }
      return _pq.top();
    }

private:
  int priority;
  priority_queue<int, vector<int>, greater<int>> _pq;
};

class TicTacToe {
public:
    /** Initialize your data structure here. */
    TicTacToe(int n): rows(n), cols(n), N(n), diag(0), rev_diag(0) {}

    int move(int row, int col, int player) {
        int add = player == 1 ? 1 : -1;
        rows[row] += add;
        cols[col] += add;
        diag += (row == col ? add : 0);
        rev_diag += (row == N - col - 1 ? add : 0);
        return (abs(rows[row]) == N || abs(cols[col]) == N || abs(diag) == N || abs(rev_diag) == N) ? player : 0;
    }

private:
    vector<int> rows, cols;
    int diag, rev_diag, N;
};

class NestedInteger {
public:
  // Return true if this NestedInteger holds a single integer, rather than a nested list.
  bool isInteger() const;

  // Return the single integer that this NestedInteger holds, if it holds a single integer
  // The result is undefined if this NestedInteger holds a nested list
  int getInteger() const;

  // Return the nested list that this NestedInteger holds, if it holds a nested list
  // The result is undefined if this NestedInteger holds a single integer
  const vector<NestedInteger> &getList() const;
};

//https://leetcode.com/problems/flatten-nested-list-iterator/
class NestedIterator {
public:
  NestedIterator(const vector<NestedInteger> &nestedList) {
    flatten(nestedList);
  }

  int next() {
    int n = st.top().getInteger();
    st.pop();
    return n;
  }

  bool hasNext() {
    while (!st.empty() && !st.top().isInteger()) {
      auto list = st.top().getList();
      st.pop();
      flatten(list);
    }
    return !st.empty();
  }

private:
  void flatten(vector<NestedInteger> list) {
    for (int i = list.size() - 1; i >= 0; i--) {
      st.emplace(list[i]);
    }
  }

  stack<NestedInteger> st;
};

//https://leetcode.com/problems/shuffle-an-array/
class Shuffle {
public:
  Shuffle(vector<int> &nums) {
    this->data = nums;
  }

  /** Resets the array to its original configuration and return it. */
  vector<int> reset() {
    return this->data;
  }

  /** Returns a random shuffling of the array. */
  vector<int> shuffle() {
    vector<int> res(this->data);
    for (int i = 0; i < res.size(); i++) {
      int j = this->e() % res.size();
      swap(res[i], res[j]);
    }
    return res;
  }

private:
  vector<int> data;
  default_random_engine e;
};

class Solution {
public:
  TreeNode* sortedArrayToBST(vector<int>& nums) {
    return sortedArrayToBSTHelper(nums, 0, nums.size());
  }

  TreeNode* sortedArrayToBSTHelper(vector<int>& nums, int start, int end) {
    if (start >= end) {
      return nullptr;
    }
    auto mid = (start + end) / 2;
    auto *root = new TreeNode(nums[mid]);
    root->left = sortedArrayToBSTHelper(nums, start, mid);
    root->right = sortedArrayToBSTHelper(nums, mid+1, end);
    return root;
  }

//  int firstUniqChar(string s) {
//    if (s.empty()) {
//      return -1;
//    }
//    vector<int> res = vector<int>(256, 0);
//    for (int i = 0; i < s.size(); ++i) {
//      res[(int)s[i]] += 1;
//    }
//
//    for (int j = 0; j < s.size(); ++j) {
//      if (res[(int)s[j]] == 1) {
//        return j;
//      }
//    }
//    return -1;
//  }

  bool isPowerOfThree(int n) {
    return (n > 0) && (1162261467 % n == 0);
  }


//  vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
//    vector<vector<int>> res {};
//    vector<int> mid {};
//    combinationSumHelper(candidates, res, target, mid, 0);
//    return res;
//  }
//  void combinationSumHelper(vector<int> candidates, vector<vector<int>>& res,
//      int target, vector<int>& mid, int start) {
//    if (target == 0) {
//      res.push_back(mid);
//      return;
//    }
//
//    for (int i = start; i < candidates.size(); ++i) {
//      if (target < candidates[i]) {
//        continue;
//      }
//      mid.push_back(candidates[i]);
//      combinationSumHelper(candidates,res,target - candidates[i],mid, i);
//      mid.pop_back();
//    }
//  }

  int findTargetSumWays(vector<int>& nums, int S) {
    int sum = 0;
    for (auto num : nums) {
      sum += num;
    }
    if (S > sum || S < -sum || ((S + sum) & 1) == 1) {
      return 0;
    }
    return sumTarget(nums, (S + sum) >> 1);
  }

  int sumTarget(vector<int>& nums, int target) {
    vector<int> res(target + 1, 0);
    res[0] = 1;

    for (int i = 0; i < nums.size(); ++i) {
      for (int j = target; j >= nums[i]; --j) {
        res[j] += res[j - nums[i]];
      }
    }
    return res[target];
  }

  //https://leetcode.com/problems/number-of-islands/
  int numIslands(vector<vector<char>>& grid) {
    if (grid.empty()) return 0;

    int res = 0;
    for (int i = 0; i < grid.size(); ++i) {
      for (int j = 0; j < grid[0].size(); ++j) {
        if (grid[i][j] == '1') {
          numIslandsHelper(grid, i, j, grid.size(), grid[0].size());
          ++res;
        }
      }
    }
    return res;
  }
  void numIslandsHelper(vector<vector<char>>& grid, int i, int j, int row, int col) {
    if (i < 0 || j < 0 || i >= row || j >= col || grid[i][j] != '1') return;
    grid[i][j] = '0';
    numIslandsHelper(grid, i-1, j, row, col);
    numIslandsHelper(grid, i+1, j, row, col);
    numIslandsHelper(grid, i, j-1, row, col);
    numIslandsHelper(grid, i, j+1, row, col);
  }

  //https://leetcode.com/problems/max-area-of-island/
  int maxAreaOfIsland(vector<vector<int>>& grid) {
    if (grid.empty()) return 0;
    int res = 0, row = grid.size(), col = grid[0].size();
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        if (grid[i][j] == 1) {
          int area = getLandArea(grid, i, j, row, col);
          res = std::max(res, area);
        }
      }
    }
    return res;
  }
  int getLandArea(vector<vector<int>>& grid, int i, int j, int row, int col) {
    if (i < 0 || j < 0 || i >= row || j >= col || grid[i][j] == 0) return 0;
    grid[i][j] = 0;
    return 1 + getLandArea(grid, i, j-1, row, col) + getLandArea(grid, i-1, j, row, col) +
    getLandArea(grid, i, j+1, row, col) + getLandArea(grid, i+1, j, row, col);
  }

  //https://leetcode.com/problems/island-perimeter/
  int islandPerimeter(vector<vector<int>>& grid) {
    if (grid.empty()) return 0;

    int res = 0, row = grid.size(), col = grid[0].size();
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        if (grid[i][j] == 1) {
          res += (4 - isLand(grid, i-1, j, row, col) - isLand(grid, i, j-1, row, col)
            - isLand(grid, i+1, j, row, col) - isLand(grid, i, j+1, row, col));
        }
      }
    }
    return res;
  }

  int isLand(vector<vector<int>>& grid, int i, int j, int row, int col) {
    if (i < 0 || j < 0 || i >= row || j >= col) return 0;
    return grid[i][j];
  }

  //https://leetcode.com/problems/flood-fill/
  vector<vector<int>> floodFill(vector<vector<int>>& image, int sr, int sc, int newColor) {
    if (image.empty() || sr >= image.size() || sc >= image[0].size() || image[sr][sc] == newColor) return image;
    fillAround(image, sr, sc, image.size(), image[0].size(), image[sr][sc], newColor);
    return image;
  }
  void fillAround(vector<vector<int>>& image, int i, int j, int row, int col, int sColor, int newColor) {
    if (i < 0 || j < 0 || i >= row || j >= col || image[i][j] != sColor) return;
    image[i][j] = newColor;
    fillAround(image, i-1, j, row, col, sColor, newColor);
    fillAround(image, i, j-1, row, col, sColor, newColor);
    fillAround(image, i+1, j, row, col, sColor, newColor);
    fillAround(image, i, j+1, row, col, sColor, newColor);
  }

  //https://leetcode.com/problems/coloring-a-border/
  vector<vector<int>> colorBorder(vector<vector<int>>& grid, int r0, int c0, int color) {
    if (grid.empty()) return grid;
    fill(grid, r0, c0, grid.size(), grid[0].size(), grid[r0][c0]);
    for (int i = 0; i < grid.size(); i++) {
      for (int j = 0; j < grid[0].size(); j++) {
        if (grid[i][j] < 0) grid[i][j] = color;
      }
    }
    return grid;
  }

  void fill(vector<vector<int>>& grid, int i, int j, int row, int col, int color) {
    if (i < 0 || j < 0 || i >= row || j >= col || grid[i][j] != color) return;
    grid[i][j] = -color;
    fill(grid, i, j-1, row, col, color);
    fill(grid, i-1, j, row, col, color);
    fill(grid, i, j+1, row, col, color);
    fill(grid, i+1, j, row, col, color);
    if (i > 0 && i < row-1 && j > 0 && j < col-1 &&
      std::abs(grid[i-1][j]) == color &&
      std::abs(grid[i][j-1]) == color &&
      std::abs(grid[i+1][j]) == color &&
      std::abs(grid[i][j+1]) == color) {
      grid[i][j] = color;
    }
  }

  //https://leetcode.com/problems/number-of-distinct-islands/
  int numDistinctIslands(vector<vector<int>>& grid) {
    if (grid.empty()) return 0;
    int row = grid.size(), col = grid[0].size();
    unordered_set<string> lands{};
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        if (grid[i][j] == 1) {
          vector<char> path{};
          dfs(grid, i, j, row, col, path, '0');
          lands.emplace(string(path.begin(), path.end()));
        }
      }
    }
    return lands.size();
  }
  void dfs(vector<vector<int>>& grid, int i, int j, int row, int col, vector<char>& path, char dir) {
    if (i < 0 || i >= row || j < 0 || j >= col || grid[i][j] != 1) return;
    grid[i][j] = 0;
    path.emplace_back(dir);
    dfs(grid, i, j-1, row, col, path, '1');
    dfs(grid, i-1, j, row, col, path, '2');
    dfs(grid, i, j+1, row, col, path, '3');
    dfs(grid, i+1, j, row, col, path, '4');
  }

  //https://leetcode.com/problems/surrounded-regions/
  void solve(vector<vector<char>>& board) {
    if (board.empty() || board[0].empty()) return;
    int row = board.size(), col = board[0].size();
    for (int i = 0; i < col; i++) {
      if (board[0][i] == 'O') {
        change(board, 0, i, row, col);
      }
    }
    for (int i = 0; i < row; i++) {
      if (board[i][col-1] == 'O') {
        change(board, i, col-1, row, col);
      }
    }
    for (int i = col-1; i >= 0; i--) {
      if (board[row-1][i] == 'O') {
        change(board, row-1, i, row, col);
      }
    }
    for (int i = row-1; i >= 0; i--) {
      if (board[i][0] == 'O') {
        change(board, i, 0, row, col);
      }
    }

    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        if (board[i][j] == 'O') {
          board[i][j] = 'X';
        } else if (board[i][j] == 'V') {
          board[i][j] = 'O';
        }
      }
    }
  }
  void change(vector<vector<char>>& image, int i, int j, int row, int col) {
    if (i < 0 || j < 0 || i >= row || j >= col || image[i][j] != 'O') return;
    image[i][j] = 'V';
    change(image, i-1, j, row, col);
    change(image, i, j-1, row, col);
    change(image, i+1, j, row, col);
    change(image, i, j+1, row, col);
  }

  int subarraySum(vector<int>& nums, int k) {
    std::unordered_map<int, int> map {};
    map[0] = 1;
    int sum = 0;
    int count = 0;
    for (auto num : nums) {
      sum += num;
      if (map.find(sum - k) != map.end()) {
        count += map[sum - k];
      }
      map[sum] += 1;
    }
    return count;
  }

  string minWindow(string s, string t) {
    if (s.empty() || t.empty() || s.size() < t.size())
      return "";

    std::unordered_map<char, int> m{};
    for (auto chr : t) {
      if (m.find(chr) != m.end()) {
        m[chr] += 1;
      } else {
        m[chr] = 1;
      }
    }

    int left = 0, right = 0, counter = t.size();
    int m_left = 0, m_right = s.size();
    bool isMatch = false;
    while (right < s.size()) {
      if ((m.find(s[right]) != m.end()) && m[s[right]] > 0) {
        counter--;
      }
      m[s[right]] -= 1;
      right++;

      while (counter == 0) {
        isMatch = true;
        if (right - left <= m_right - m_left) {
          m_right = right;
          m_left = left;
        }

        if ((m.find(s[left]) != m.end()) && m[s[left]] >= 0) {
          counter++;
        }
        m[s[left]] += 1;
        left++;
      }
    }

    return isMatch ? s.substr(m_left, m_right - m_left) : "";
  }

  vector<int> findAnagrams(string s, string p) {
    if (s.empty() || p.empty() || s.size() < p.size()) {
      return vector<int>{};
    }

    std::unordered_map<char, int> m{};
    for (auto chr : p) {
      m[chr] += 1;
    }

    int left = 0, right = 0, counter = m.size();
    vector<int> res{};
    while (right < s.size()) {
      if (m.find(s[right]) != m.end()) {
        m[s[right]] -= 1;
        if (m[s[right]] == 0) counter--;
      }
      right++;

      while (counter == 0) {
        if (right - left == p.size()) {
          res.emplace_back(left);
        }

        if (m.find(s[left]) != m.end()) {
          m[s[left]] += 1;
          if (m[s[left]] > 0) counter++;
        }
        left++;
      }
    }
    return res;
  }

  int lengthOfLongestSubstring(string s) {
//    std::unordered_map<char, int> m{};
//    int res = 0;
//    for (int i = 0; i < s.size(); i++) {
//      if (m.find(s[i]) == m.end()) {
//        m[s[i]] = i;
//        if (m.size() > res) res = m.size();
//      } else {
//        i = m[s[i]];
//        m.clear();
//      }
//    }
//    return res;

//    std::unordered_map<char, int> m{};
//    int res = 0, start = -1;
//    for (int i = 0; i < s.size(); ++i) {
//      if (m.find(s[i]) != m.end()) {
//        start = std::max(m[s[i]], start);
//      }
//      m[s[i]] = i;
//      res = std::max(res, i - start);
//    }
//    return res;

    std::unordered_map<char, int> m{};
    int start = 0, end = 0, counter = 0, res = 0;
    while (end < s.size()) {
      m[s[end]] += 1;
      if (m[s[end]] > 1) counter++;
      end++;

      while (counter > 0) {
        m[s[start]] -= 1;
        if (m[s[start]] > 0) counter--;
        start++;
      }
      res = std::max(res, end - start);
    }
    return res;
  }

  vector<int> findSubstring(string s, vector<string>& words) {
    if (s.empty() || words.empty() || s.size() < (words.size() * words[0].size())) return vector<int>{};
    std::unordered_map<string, int> m{}, mid{};
    for (const string& word : words) {
      m[word] += 1;
    }

    int start = 0, end = words[0].size() * words.size() - 1, once = words[0].size(), counter = m.size();
    while (end < s.size()) {
      for (int i = 0; i < end; i += once) {
        if (m.find(s.substr(i, once)) != m.end()) {

        }
      }
    }
  }

//
//  int countPrimes(int n) {
//    int count = 0;
//    bool not_prime[n];
//    for (int i = 2; i < n; ++i) {
//      if (!not_prime[i]) {
//        count++;
//        for (int j = 2; j * i < n; j++) {
//          not_prime[i * j] = true;
//        }
//      }
//    }
//    return count;
//  }

  //https://leetcode.com/problems/3sum/
  vector<vector<int>> threeSum(vector<int>& nums) {
    std::sort(nums.begin(), nums.end());
    int len = nums.size();
    vector<vector<int>> res{};
    for (int i = 0; i < len; ++i) {
      if ((i - 1) >= 0 && nums[i] == nums[i - 1]) continue;
      int l = i + 1, r = len - 1;
      while (l < r) {
        int sum = nums[i] + nums[l] + nums[r];
        if (sum > 0) {
          r--;
        } else if (sum < 0) {
          l++;
        } else {
          res.emplace_back(vector<int>{nums[i], nums[l], nums[r]});
          l++, r--;
          while ((l < r) && (nums[l] == nums[l - 1])) l++;
          while ((l < r) && nums[r] == nums[r + 1]) r--;
        }
      }
    }
    return res;
  }

  //O(n^3), TimeLimit
  void threeSumHelper(vector<int>& nums, vector<vector<int>>& res, vector<int>& tmp_list, int begin) {
    if (tmp_list.size() == 3) {
      if ((tmp_list[0] + tmp_list[1] + tmp_list[2]) == 0) res.emplace_back(tmp_list);
    } else {
      for (int i = begin; i < nums.size(); i++) {
        if (i > begin && nums[i] == nums[i-1]) continue;
        tmp_list.emplace_back(nums[i]);
        threeSumHelper(nums, res, tmp_list, i+1);
        tmp_list.pop_back();
      }
    }
  }
  vector<vector<int>> threeSum2(vector<int>& nums) {
    std::sort(nums.begin(), nums.end());
    vector<vector<int>> res{};
    vector<int> tmp_list{};
    threeSumHelper(nums, res, tmp_list, 0);
    return res;
  }

  //https://leetcode.com/problems/3sum-closest/
  int threeSumClosest(vector<int>& nums, int target) {
    std::sort(nums.begin(), nums.end());
    int len = nums.size(), gap = std::numeric_limits<int>::max(), res = 0;
    for (int i = 0; i < len - 2; i++) {
      int l = i + 1, r = len - 1;
      while (l < r) {
        int sum = nums[i] + nums[l] + nums[r];
        if (std::abs(sum - target) < gap) {
          gap = std::abs(sum - target);
          res = sum;
        }

        if (sum > target) {
          r--;
        } else if (sum < target) {
          l++;
        } else {
          return target;
        }
      }
    }
    return res;
  }

  //https://leetcode.com/problems/3sum-smaller/
  int threeSumSmaller(vector<int> &nums, int target) {
    std::sort(nums.begin(), nums.end());
    int res = 0, n = nums.size();
    for (int i = 0; i < n - 2; ++i) {
      int left = i + 1, right = n - 1;
      while (left < right) {
        if (nums[i] + nums[left] + nums[right] < target) {
          res += right - left;
          ++left;
        } else {
          --right;
        }
      }
    }
    return res;
  }

  //https://leetcode.com/problems/4sum/
  vector<vector<int>> fourSum(vector<int>& nums, int target) {
    std::sort(nums.begin(), nums.end());
    vector<vector<int>> res{};
    int left, right, sum, n = nums.size();
    for (int i = 0; i < n - 3; i++) {
      if (i > 0 && nums[i] == nums[i-1]) continue;

      for (int j = i + 1; j < n - 2; j++) {
        if (j > i+1 && nums[j] == nums[j-1]) continue;

        left = j + 1, right = n - 1;
        while (left < right) {
          sum = nums[i] + nums[j] + nums[left] + nums[right];
          if (sum > target) {
            right--;
          } else if (sum < target) {
            left++;
          } else {
            res.emplace_back(std::move(vector<int>{nums[i], nums[j], nums[left], nums[right]}));
            left++, right--;
            while (left < right && nums[left] == nums[left-1]) left++;
            while (left < right && nums[right] == nums[right+1]) right--;
          }
        }
      }
    }
    return res;
  }

  //https://leetcode.com/problems/4sum-ii/
  int fourSumCount(vector<int> &A, vector<int> &B, vector<int> &C, vector<int> &D) {
    std::unordered_map<int, int> m{};
    for (int i = 0; i < A.size(); i++) {
      for (int j = 0; j < B.size(); j++) {
        m[A[i] + B[j]] += 1;
      }
    }

    int res = 0;
    for (int i = 0; i < C.size(); i++) {
      for (int j = 0; j < D.size(); j++) {
        res += m[0 - C[i] - D[j]];
      }
    }
    return res;
  }

  int threeSumMulti(vector<int>& A, int target) {
    std::sort(A.begin(), A.end());
    int count = 0, len = A.size();
    for (int i = 0; i < len - 2; i++) {
      int l = i + 1, r = len - 1;
      while (l < r) {
        int sum = A[i] + A[l] + A[r];
        if (sum > target) {
          r--;
        } else if (sum < target) {
          l++;
        } else {
          int l_start = l, r_start = r;
          cout << i << ", " << l << ", " << r << endl;
          cout << "A[i]=" << A[i] << ", A[l]=" << A[l] << ", A[r]=" << A[r] << endl;
          l++, r--;
          while ((l <= r) && (A[l] == A[l - 1])) l++;
          while ((l <= r) && (A[r] == A[r + 1])) r--;
          cout << (l - l_start) * (r_start - r) << endl;
          cout << "l=" << l << " r=" << r << endl;
          if (l == r_start) {
            count += (r_start - l_start + 1) * (r_start - l_start) / 2;
          } else {
            count += (l - l_start) * (r_start - r);
          }
        }
      }
    }
    return count;
  }

  int maxDepth(TreeNode* root) {
    if (root == nullptr) return 0;
    return 1 + max(maxDepth(root->left), maxDepth(root->right));
  }

  int singleNumber(vector<int>& nums) {
    int res = 0;
    for (int num : nums) {
      res ^= num;
    }
    return res;
  }

  ListNode* reverseList(ListNode* head) {
    ListNode res(0);
    while (head) {
      ListNode *tail = res.next;
      res.next = head;
      head = head->next;
      res.next->next = tail;
    }
    return res.next;
  }

  int majorityElement(vector<int>& nums) {
    int count = 0, res = nums[0];
    for(int num : nums) {
      if (res == num) {
        count++;
      } else {
        count--;
      }
      if (count == 0) {
        res = num;
        count = 1;
      }
    }
    return res;
  }

  //https://leetcode.com/problems/move-zeroes/
  void moveZeroes(vector<int>& nums) {
    int start = 0;
    while ((start < nums.size()) && (nums[start] != 0)) start++;
    for (int i = start; i < nums.size(); i++) {
      if (nums[i] != 0) {
        nums[start++] = nums[i];
        nums[i] = 0;
      }
    }
  }

  //https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/
  int maxProfit(vector<int>& prices) {
    if (prices.size() < 2) return 0;
    int profit = 0;
    for (int i = 1; i < prices.size(); i++) {
      if (prices[i] - prices[i-1] > 0) {
        profit += (prices[i] - prices[i-1]);
      }
    }
    return profit;
  }

  //https://leetcode.com/problems/valid-anagram/
  bool isAnagram(string s, string t) {
    if (s.size() != t.size()) return false;
    unordered_map<char, int> m{};
    for (int i = 0; i < s.size(); i++) {
      m[s[i]]++;
      m[t[i]]--;
    }
    for (auto v : m) {
      if (v.second) return false;
    }
    return true;

//    if (s.size() != t.size()) return false;
//    unordered_map<char, int> m{};
//    for (auto chr : s) {
//      m[chr] += 1;
//    }
//    for (auto chr: t) {
//      if ((m.find(chr) != m.end()) && (m[chr] > 0)) {
//        m[chr] -= 1;
//      } else {
//        return false;
//      }
//    }
//    return true;
  }

  //https://leetcode.com/problems/contains-duplicate/
  bool containsDuplicate(vector<int>& nums) {
    std::unordered_set<int> s{};
    for (auto num : nums) {
      if (s.find(num) != s.end()) {
        return true;
      } else {
        s.insert(num);
      }
    }
    return false;
  }

  int titleToNumber(string s) {
    if (s.empty()) return 0;
    int sum = 0;
    for (int i = 0; i < s.size(); i++) {
      sum = (s[i] - 'A' + 1) + sum * 26;
    }
    return sum;
  }

  //https://leetcode.com/problems/first-unique-character-in-a-string/
  int firstUniqChar(string s) {
    if (s.empty()) return -1;
    std::unordered_map<char, int> m{};
    for (auto chr : s) {
      m[chr] += 1;
    }
    for (int i = 0; i < s.size(); i++) {
      if (m[s[i]] == 1) return i;
    }
    return -1;
  }

  //https://leetcode.com/problems/merge-two-sorted-lists/
  ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
    ListNode head{0};
    ListNode* cur = &head;
    while (l1 != nullptr && l2 != nullptr) {
      if (l1->val < l2->val) {
        cur->next = l1;
        l1 = l1->next;
        cur = cur->next;
      } else {
        cur->next = l2;
        l2 = l2->next;
        cur = cur->next;
      }
    }
    if (l1 != nullptr) cur->next = l1;
    if (l2 != nullptr) cur->next = l2;
    return head.next;
  }

  //https://leetcode.com/problems/reverse-bits/
  uint32_t reverseBits(uint32_t n) {
    uint32_t res = 0;
    for (int i = 0; i < 32; i++) {
      res = (n & 1) + (res << 1);
      n = n >> 1;
    }
    return res;
  }

  //https://leetcode.com/problems/reverse-string/
  void reverseString(vector<char>& s) {
    int start = 0, end = s.size() - 1;
    while (end > start) {
      auto tmp = s[start];
      s[start] = s[end];
      s[end] = tmp;
      end--;
      start++;
    }
  }

  //https://leetcode.com/problems/pascals-triangle/
  vector<vector<int>> generate(int numRows) {
    vector<vector<int>> res(numRows);
    for (int i = 0; i < numRows; i++) {
      vector<int> once(i + 1);
      for (int j = 0; j < i + 1; j++) {
        if (j == 0 || j == i) {
          once[j] = 1;
        } else {
          once[j] = res[i-1][j-1] + res[i-1][j];
        }
      }
      res[i] = once;
    }
    return res;
  }

  //https://leetcode.com/problems/missing-number/
  int missingNumber(vector<int>& nums) {
    int res = 0;
    for (int i = 0; i < nums.size(); i++) {
      res += (i + 1);
      res -= nums[i];
    }
    return res;
  }

  //https://leetcode.com/problems/intersection-of-two-arrays-ii/
  vector<int> intersect(vector<int> &nums1, vector<int> &nums2) {
    unordered_map<int, int> m{};
    vector<int> res{};
    for (auto num : nums1) {
      m[num] += 1;
    }
    for (auto num : nums2) {
      if (m[num] > 0) {
        res.emplace_back(num);
        m[num] -= 1;
      }
    }
    return res;
  }

  //https://leetcode.com/problems/sum-of-two-integers/
  int getSum(int a, int b) {
    while (b != 0) {
      int carry = a & b;
      a = a ^ b;
      b = carry << 1;
    }
    return a;
  }

  //https://leetcode.com/problems/happy-number/
  //快慢指针, 判断环
  bool isHappy(int n) {
    std::unordered_set<int> res{n};
    int sum;
    while (true) {
      sum = 0;
      while (n != 0) {
        int rem = n % 10;
        sum += rem * rem;
        n /= 10;
      }
      if (sum == 1) return true;
      if (res.find(sum) != res.end()) return false;
      res.insert(sum);
      n = sum;
    }
  }

  //https://leetcode.com/problems/number-of-1-bits/
  int hammingWeight(uint32_t n) {
    int count = 0;
    while (n) {
      n &= (n - 1);
      count++;
    }
    return count;

//    int sum = 0;
//    while (n) {
//      sum += n & 1;
//      n = n >> 1;
//    }
//    return sum;
  }

  //https://leetcode.com/problems/climbing-stairs/
  int climbStairs(int n) {
    vector<int> dp(n + 1, 1);
    for (int i = 2; i <= n; i++) {
      dp[i] = dp[i-1] + dp[i-2];
    }
    return dp[n];
  }

  //https://leetcode.com/problems/symmetric-tree/
  bool helper(TreeNode* left, TreeNode* right) {
    if (nullptr == left && nullptr == right) return true;
    if (left != nullptr && right != nullptr) {
      return (left->val == right->val) && helper(left->left, right->right) && helper(left->right, right->left);
    }
    return false;
  }
  bool isSymmetric(TreeNode* root) {
    if (nullptr == root) return true;
    return helper(root->left, root->right);
  }

  //https://leetcode.com/problems/maximum-subarray/
  int maxSubArray(vector<int>& nums) {
    if (nums.empty()) return 0;
    int sum = 0, max = std::numeric_limits<int>::min();
    for (int i = 0; i < nums.size(); i++) {
      sum += nums[i];
      max = std::max(max, sum);
      if (sum < 0) sum = 0;
    }
    return max;
  }

  //https://leetcode.com/problems/two-sum/
  vector<int> twoSum(vector<int>& nums, int target) {
    if (nums.empty()) return vector<int>{};
    unordered_map<int, int> m{};
    for (int i = 0; i < nums.size(); i++) {
      if (m.find(nums[i]) != m.end()) {
        return vector<int>{m[nums[i]], i};
      } else {
        m[target - nums[i]] = i;
      }
    }
    return vector<int> {};
  }

  //https://leetcode.com/problems/remove-duplicates-from-sorted-array/
  int removeDuplicates(vector<int>& nums) {
    if (nums.empty()) return 0;
    int sum = 1, once = nums[0];
    for (int i = 1; i < nums.size(); i++) {
      if (once != nums[i]) {
        nums[sum] = nums[i];
        sum += 1;
        once = nums[i];
      }
    }
    return sum;
  }

  //https://leetcode.com/problems/plus-one/
  vector<int> plusOne(vector<int>& digits) {
    if (digits.empty()) return digits;
    int len = digits.size();
    vector<int> res;
    for (int i = len - 1; i >= 0; i--) {
      if (digits[i] == 9) {
        digits[i] = 0;
      } else {
        digits[i] += 1;
        return digits;
      }
    }
    digits.insert(digits.begin(), 1);
    return digits;
//    if (digits.empty()) return digits;
//    int len = digits.size();
//    for (int i = len - 1; i >= 0; i--) {
//      if (digits[i] == 9) {
//        digits[i] = 0;
//      } else {
//        digits[i] += 1;
//        return digits;
//      }
//    }
//    digits.insert(digits.begin(), 1);
//    return digits;
  }

  //https://leetcode.com/problems/house-robber/
  int rob(vector<int>& nums) {
    if (nums.empty()) return 0;
    int len = nums.size();
    if (len == 1) return nums[0];
    if (len == 2) return std::max(nums[0], nums[1]);

    vector<int> bp(len);
    bp[0] = nums[0];
    bp[1] = std::max(nums[0], nums[1]);
    for (int i = 2; i < len; i++) {
      bp[i] = std::max(bp[i-2] + nums[i], bp[i-1]);
    }
    return bp[len-1];
  }

  //https://leetcode.com/problems/linked-list-cycle/
  bool hasCycle(ListNode *head) {
    ListNode *slow, *fast;
    slow = fast = head;
    while (fast != nullptr && fast->next != nullptr) {
      slow = slow->next;
      fast = fast->next->next;
      if (slow == fast) return true;
    }
    return false;
  }

  //https://leetcode.com/problems/intersection-of-two-linked-lists/
  ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
    if (nullptr == headA || nullptr == headB) return nullptr;
    ListNode *ptr_a, *ptr_b;
    ptr_a = headA, ptr_b = headB;
    while (ptr_a != ptr_b) {
      if (ptr_a == nullptr) {
        ptr_a = headB;
      } else {
        ptr_a = ptr_a->next;
      }
      if (ptr_b == nullptr) {
        ptr_b = headA;
      } else {
        ptr_b = ptr_b->next;
      }
    }
    return ptr_a;

//    if (nullptr == headA || nullptr == headB) return nullptr;
//    int a_len = 0, b_len = 0;
//    ListNode *ptr_a = headA, *ptr_b = headB;
//    while (nullptr != ptr_a && nullptr != ptr_a->next) {
//      ptr_a = ptr_a->next;
//      a_len++;
//    }
//    while (nullptr != ptr_b && nullptr != ptr_b->next) {
//      ptr_b = ptr_b->next;
//      b_len++;
//    }
//
//    if (ptr_a != ptr_b) return nullptr;
//    if (a_len >= b_len) {
//      while (a_len > b_len) {
//        headA = headA->next;
//        a_len--;
//      }
//    } else {
//      while (a_len < b_len) {
//        headB = headB->next;
//        b_len--;
//      }
//    }
//    while (headA != headB) {
//      headA = headA->next;
//      headB = headB->next;
//    }
//    return headA;
  }

  //https://leetcode.com/problems/merge-sorted-array/
  void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
    int p = n+m-1, p1 = m-1, p2 = n-1;
    while (p2 >= 0 && p1 >= 0) {
      if (nums2[p2] > nums1[p1]) {
        nums1[p] = nums2[p2];
        p2--;
      } else {
        nums1[p] = nums1[p1];
        p1--;
      }
      p--;
    }
    while (p2 >=0) {
      nums1[p] = nums2[p2];
      p2--, p--;
    }
  }

  //https://leetcode.com/problems/palindrome-linked-list/
  bool isPalindrome(ListNode* head) {
    ListNode *fast = head, *slow = head, *pre = nullptr;
    while (nullptr != fast && nullptr != fast->next) {
      fast = fast->next->next;
      //slow to next and reverse
      ListNode *next = slow->next;
      slow->next = pre;
      pre = slow;
      slow = next;
    }

    if (nullptr != fast) slow = slow->next;
    while (slow) {
      if (pre->val != slow->val) return false;
      slow = slow->next;
      pre = pre->next;
    }
    return true;

//    ListNode *fast = head, *slow = head;
//    while (nullptr != fast && nullptr != fast->next) {
//      slow = slow->next;
//      fast = fast->next->next;
//    }
//    if (nullptr != fast) slow = slow->next;
//
//    //reverse
//    ListNode node(0);
//    ListNode *mid;
//    mid = &node;
//    while (slow) {
//      ListNode *tmp = slow->next;
//      slow->next = mid->next;
//      mid->next = slow;
//      slow = tmp;
//    }
//
//    while (mid->next) {
//      if (head->val != mid->next->val) {
//        return false;
//      }
//      head = head->next;
//      mid = mid->next;
//    }
//    return true;

//    vector<int> dp{};
//    while (head) {
//      dp.emplace_back(head->val);
//      head = head->next;
//    }
//
//    int len = dp.size(), mid = (len >> 1);
//    for (int i = 0; i < mid; i++) {
//      if (dp[i] != dp[len - i - 1]) {
//        return false;
//      }
//    }
//    return true;
  }

  //https://leetcode.com/problems/valid-parentheses/
  bool isValid(string s) {
    std::stack<char> st{};
    for (auto ch : s) {
      switch (ch) {
        case '(':
        case '[':
        case '{':
          st.push(ch);
          break;
        case ')':
          if (st.empty() || st.top() != '(') return false;
          st.pop();
          break;
        case ']':
          if (st.empty() || st.top() != '[') return false;
          st.pop();
          break;
        case '}':
          if (st.empty() || st.top() != '{') return false;
          st.pop();
          break;
        default:
          return false;
      }
    }
    return st.empty();
  }

  //https://leetcode.com/problems/factorial-trailing-zeroes/
  int trailingZeroes(int n) {
    int res = 0;
    while (n) {
      n /= 5;
      res += n;
    }
    return res;
  }

  //https://leetcode.com/problems/valid-palindrome/
  bool isAlphaChar(char ch) {
    return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9');
  }
  bool isPalindrome(string s) {
    if (s.empty()) return true;
    int start = 0, end = s.size() - 1;
    while (end > start) {
      while ((end > start) && !isAlphaChar(s[start])) {
        start++;
      }
      while ((end > start) && !isAlphaChar(s[end])) {
        end--;
      }
      if (end > start) {
        if (s[start] == s[end] || (min(s[start], s[end]) > '9' && abs(s[start] - s[end]) == 32)) {
          start++;
          end--;
        } else {
          return false;
        }
      }
    }
    return true;
  }

  //https://leetcode.com/problems/longest-common-prefix/
  string longestCommonPrefix(vector<string>& strs) {
    if (strs.empty()) return "";
    int idx = 0;
    while (idx < strs[0].size()) {
      for (int i = 1; i < strs.size(); i++) {
        if (idx >= strs[i].size() || strs[0][idx] != strs[i][idx]) return strs[0].substr(0, idx);
      }
      idx++;
    }
    return strs[0].substr(0, idx);
  }

  //https://leetcode.com/problems/rotate-array/
  void reverse(vector<int>& nums, int start, int end) {
    while (start < end) {
      int tmp = nums[start];
      nums[start] = nums[end];
      nums[end] = tmp;
      start++;
      end--;
    }
  }
  void rotate(vector<int>& nums, int k) {
    if (nums.empty()) return;
    int len = nums.size();
    k %= len;
    // BA = ((BA)^T)^T = (A^TB^T)^T, transform
    reverse(nums, 0, len-k-1);
    reverse(nums, len-k, len-1);
    reverse(nums, 0, len-1);
  }

  //https://leetcode.com/problems/implement-strstr/
  int KMP(string haystack, string needle) {
    if (needle.empty()) return 0;

    //KMP
    int pat_size = needle.size(), sim_state = 0;
    vector<vector<int>> dp(pat_size, vector<int>(256, 0));
    dp[0][needle[0]] = 1;
    for (int i = 1; i < pat_size; i++) {
      for (int j = 0; j < 256; j++) {
        if (needle[i] == j) {
          dp[i][j] += 1;
        } else {
          dp[i][j] = dp[sim_state][j];
        }
      }
      sim_state = dp[sim_state][needle[i]];
    }

    //search
    int txt_size = haystack.size();
    sim_state = 0;
    for (int i = 0; i < txt_size; i++) {
      sim_state = dp[sim_state][haystack[i]];
      if (sim_state == pat_size) {
        return i - pat_size + 1;
      }
    }
    return -1;
  }

  int strStr(string haystack, string needle) {
    if (needle.empty()) return 0;

    //KMP
    int pat_size = needle.size(), sim_state = 0;
    vector<vector<int>> dp(pat_size, vector<int>(256, 0));
    dp[0][needle[0]] = 1;
    for (int i = 1; i < pat_size; i++) {
      for (int j = 0; j < 256; j++) {
        if (needle[i] == j) {
          dp[i][j] = i + 1;
        } else {
          dp[i][j] = dp[sim_state][j];
        }
      }
      sim_state = dp[sim_state][needle[i]];
    }

    //search
    int txt_size = haystack.size();
    sim_state = 0;
    for (int i = 0; i < txt_size; i++) {
      sim_state = dp[sim_state][haystack[i]];
      if (sim_state == pat_size) {
        return i - pat_size + 1;
      }
    }
    return -1;

//    if (needle.empty()) return 0;
//    if (haystack.size() < needle.size()) return -1;
//    int m = haystack.size(), n = needle.size();
//    for (int i = 0; i <= m - n; i++) {
//      if (haystack.substr(i, n) == needle) {
//        return i;
//      }
//    }
//    return -1;
  }

  //https://leetcode.com/problems/sqrtx/
  int mySqrt(int x) {
    double x0 = x * 1.0 / 2, limit = 0.0000001;
    while (abs(x0 - x / x0) > limit) {
      x0 = (x0 + x / x0) / 2;
    }
    return int(x0);


//    if (x == 1)
//      return 1;
//    int start = 0, end = x;
//    while (start != end) {
//      int mid = (start + end) >> 1;
//      int f = mid * mid - x;
//      if ((f == 0) || (mid == start)) {
//        return mid;
//      } else if (f * (end * end - x) < 0) {
//        start = mid;
//      } else {
//        end = mid;
//      }
//    }
//    return start;
  }

  //https://leetcode.com/problems/count-primes/
  int countPrimes(int n) {
    int count = 0;
    vector<int> dp(n, 0);
    for (int i = 2; i < n; i++) {
      if (dp[i] == 0) {
        for (int j = 2; j*i < n; j++) {
          dp[i*j] = 1;
        }
        count++;
      }
    }
    return count;
  }

  //https://leetcode.com/problems/reverse-integer/
  int reverse(int x) {
    int abs_x = std::abs(x), res = 0;
    while (abs_x > 0) {
      if (res > (std::numeric_limits<int>::max() - abs_x % 10) / 10) {
        return 0;
      }
      res = res * 10 + abs_x % 10;
      abs_x /= 10;
    }
    return x > 0 ? res : -res;
  }

  //https://leetcode.com/problems/permutations/
  void permute_helper(vector<int> nums, int begin, vector<vector<int>>& res) {
    if (begin == nums.size()) {
      res.emplace_back(nums);
    } else {
      for (int i = begin; i < nums.size(); i++) {
        std::swap(nums[begin], nums[i]);
        permute_helper(nums, begin+1, res);
        std::swap(nums[begin], nums[i]);
      }
    }
  }
  vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> res{};
    permute_helper(nums, 0, res);
    return res;
  }

  //https://leetcode.com/problems/permutations-ii/
  bool is_swapped(vector<int>& nums, int from, int to) {
    for (int i = from; i < to; i++) {
      if (nums[to] == nums[i]) return true;
    }
    return false;
  }

  void permute_unique_helper(vector<int>& nums, int begin, vector<vector<int>>& res) {
    if (begin == nums.size()) {
      res.emplace_back(nums);
    } else {
      for (int i = begin; i < nums.size(); i++) {
        if (i > begin && is_swapped(nums, begin, i)) continue;
        std::swap(nums[begin], nums[i]);
        permute_unique_helper(nums, begin+1, res);
        std::swap(nums[begin], nums[i]);
      }
    }
  }

  vector<vector<int>> permuteUnique(vector<int>& nums) {
    vector<vector<int>> res{};
    permute_unique_helper(nums, 0, res);
    return res;
  }

  void permute_unique_helper2(vector<int>& nums, vector<int>& tmp, vector<vector<int>>& res,
    vector<bool> used) {
    if (tmp.size() == nums.size()) {
      res.emplace_back(tmp);
    } else {
      for (int i = 0; i < nums.size(); i++) {
        if (used[i] || (i > 0 && nums[i] == nums[i-1] && !used[i-1])) continue;
        used[i] = true;
        tmp.emplace_back(nums[i]);
        permute_unique_helper2(nums, tmp, res, used);
        used[i] = false;
        tmp.pop_back();
      }
    }
  }
  vector<vector<int>> permuteUnique2(vector<int>& nums) {
    vector<vector<int>> res{};
    vector<int> tmp{};
    vector<bool> used(nums.size(), false);
    std::sort(nums.begin(), nums.end());
    permute_unique_helper2(nums, tmp, res, used);
    return res;
  }


  //https://leetcode.com/problems/subsets/
  void subsets_helper(vector<int>& nums, vector<vector<int>>& res, int begin, int end, vector<int>& tmp) {
    res.emplace_back(tmp);
    for (int i = begin; i < end; i++) {
      tmp.emplace_back(nums[i]);
      subsets_helper(nums, res, i+1, end, tmp);
      tmp.pop_back();
    }
  }
  vector<vector<int>> subsets(vector<int>& nums) {
    vector<int> tmp{};
    vector<vector<int>> res{};
    subsets_helper(nums, res, 0, nums.size(), tmp);
    return res;
  }

  //https://leetcode.com/problems/subsets-ii/
  void subsets_dup_helper(vector<int>& nums, int begin, int end, vector<int>& tmp,
    vector<vector<int>>& res) {
    res.emplace_back(tmp);
    for (int i = begin; i < end; i++) {
      if (i > begin && nums[i] == nums[i-1]) continue;
      tmp.emplace_back(nums[i]);
      subsets_dup_helper(nums, i+1, end, tmp, res);
      tmp.pop_back();
    }
  }
  vector<vector<int>> subsetsWithDup(vector<int>& nums) {
    vector<vector<int>> res{};
    vector<int> tmp{};
    std::sort(nums.begin(), nums.end());
    subsets_dup_helper(nums, 0, nums.size(), tmp, res);
    return res;
  }

  //https://leetcode.com/problems/combination-sum/
  void combination_helper(vector<int>& candidates, int target, int sum,
    int start, vector<vector<int>>& res, vector<int>& tmp) {
    if (sum > target) {
      return;
    } else if (sum == target) {
      res.emplace_back(tmp);
    } else {
      for (int i = start; i < candidates.size(); i++) {
        tmp.emplace_back(candidates[i]);
        combination_helper(candidates, target, sum + candidates[i], i, res, tmp);
        tmp.pop_back();
      }
    }
  }
  vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<vector<int>> res{};
    vector<int> tmp{};
    combination_helper(candidates, target, 0, 0, res, tmp);
    return res;
  }

  //https://leetcode.com/problems/combination-sum-ii/
  void combination_helper2(vector<int>& candidates, int target, int sum, int begin, vector<int>& tmp, vector<vector<int>>& res) {
    if (sum > target) {
      return;
    } else if (sum == target) {
      res.emplace_back(tmp);
    } else {
      for (int i = begin; i < candidates.size(); i++) {
        if (i > begin && candidates[i] == candidates[i-1]) continue;
        tmp.emplace_back(candidates[i]);
        combination_helper2(candidates, target, sum + candidates[i], i+1, tmp, res);
        tmp.pop_back();
      }
    }
  }
  vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
    vector<vector<int>> res{};
    vector<int> tmp{};
    std::sort(candidates.begin(), candidates.end());
    combination_helper2(candidates, target, 0, 0, tmp, res);
    return res;
  }

  //https://leetcode.com/problems/palindrome-partitioning/
  bool is_palindrome(string s, int begin, int end) {
    while (begin < end) {
      if (s[begin] != s[end]) return false;
      begin++;
      end--;
    }
    return true;
  }

  void partition_helper(string s, int begin, vector<string>& tmp, vector<vector<string>>& res,
    vector<vector<bool>>& dp) {
    if (begin == s.size()) {
      res.emplace_back(tmp);
    } else {
      for (int i = begin; i < s.size(); i++) {
        if (dp[begin][i]) {
          tmp.emplace_back(s.substr(begin, i-begin+1));
          partition_helper(s, i+1, tmp, res, dp);
          tmp.pop_back();
        }
      }
    }
  }

  vector<vector<string>> partition(string s) {
    vector<vector<string>> res{};
    vector<string> tmp{};
    vector<vector<bool>> dp(s.size(), vector<bool>(s.size(), false));
    for (int i = 0; i < s.size(); i++) {
      for (int j = 0; j <= i; j++) {
        if (s[i] == s[j] && ((i-j <= 2) || dp[j+1][i-1])) {
          dp[j][i] = true;
        }
      }
    }
    partition_helper(s, 0, tmp, res, dp);
    return res;
  }

  //https://leetcode.com/problems/binary-tree-preorder-traversal/
  void preorder_helper(TreeNode* root, vector<int>& res) {
    if (nullptr == root) return;
    res.emplace_back(root->val);
    preorder_helper(root->left, res);
    preorder_helper(root->right, res);
  }
  vector<int> preorderTraversal(TreeNode* root) {
    vector<int> res{};
    preorder_helper(root, res);
    return res;
  }
  //iteration
  vector<int> preorderTraversal1(TreeNode* root) {
    vector<int> res{};
    stack<TreeNode*> st{};
    TreeNode* cur = root;
    while (!st.empty() || nullptr != cur) {
      while (nullptr != cur) {
        res.emplace_back(cur->val);
        st.emplace(cur);
        cur = cur->left;
      }
      cur = st.top(), st.pop();
      cur = cur->right;
    }
    return res;
  }
  //iteration2 more effective
  vector<int> preorderTraversal2(TreeNode* root) {
    if (nullptr == root) return vector<int>{};
    vector<int> res{};
    std::stack<TreeNode*> st{};
    TreeNode* cur;
    st.emplace(root);
    while (!st.empty()) {
      cur = st.top(), st.pop();
      res.emplace_back(cur->val);
      if (nullptr != cur->right) {
        st.emplace(cur->right);
      }
      if (nullptr != cur->left) {
        st.emplace(cur->left);
      }
    }
    return res;
  }
  vector<int> preorderTraversal3(TreeNode* root) {
    vector<int> res{};
    if (!root) return res;

    stack<TreeNode*> st{};
    st.emplace(root);
    while (!st.empty()) {
      TreeNode* cur = st.top();
      st.pop();
      res.emplace_back(cur->val);
      if (cur->right) st.emplace(cur->right);
      if (cur->left) st.emplace(cur->left);
    }
    return res;
  }

  //https://leetcode.com/problems/binary-tree-inorder-traversal/
  void inorder_helper(TreeNode* root, vector<int>& res) {
    if (nullptr == root) return ;
    inorder_helper(root->left, res);
    res.emplace_back(root->val);
    inorder_helper(root->right, res);
  }
  vector<int> inorderTraversal(TreeNode* root) {
    vector<int> res{};
    inorder_helper(root, res);
    return res;
  }
  //iteration
  vector<int> inorderTraversal1(TreeNode* root) {
    vector<int> res{};
    std::stack<TreeNode*> st{};
    TreeNode* cur = root;
    while (nullptr != cur || !st.empty()) {
      while (nullptr != cur) {
        st.emplace(cur);
        cur = cur->left;
      }
      cur = st.top(), st.pop();
      res.emplace_back(cur->val);
      cur = cur->right;
    }
    return res;
  }

  //https://leetcode.com/problems/binary-tree-postorder-traversal/
  void postorder_helper(TreeNode* root, vector<int>& res) {
    if (nullptr == root) return;
    postorder_helper(root->left, res);
    postorder_helper(root->right, res);
    res.emplace_back(root->val);
  }
  vector<int> postorderTraversal(TreeNode* root) {
    vector<int> res{};
    postorder_helper(root, res);
    return res;
  }
  //iteration
  vector<int> postorderTraversal1(TreeNode* root) {
    vector<int> res{};
    stack<TreeNode*> st{};
    TreeNode* cur = root;
    while (nullptr != cur || !st.empty()) {
      while (nullptr != cur) {
        res.insert(res.begin(), cur->val);
        st.emplace(cur);
        cur = cur->right;
      }
      cur = st.top(), st.pop();
      cur = cur->left;
    }
    return res;
  }
  vector<int> postorderTraversal2(TreeNode* root) {
    vector<int> res{};
    if (!root) return res;

    stack<TreeNode*> st{};
    st.emplace(root);
    while (!st.empty()) {
      TreeNode* cur = st.top();
      st.pop();
      res.insert(res.begin(), cur->val);
      if (cur->left) st.emplace(cur->left);
      if (cur->right) st.emplace(cur->right);
    }
    return res;
  }

  //https://leetcode.com/problems/validate-binary-search-tree/
  bool isValidBST_helper(TreeNode* root, long max_val, long min_val) {
    if (nullptr == root) return true;
    if (root->val >= max_val || root->val <= min_val) return false;
    return isValidBST_helper(root->left, root->val, min_val) && isValidBST_helper(root->right, max_val, root->val);
  }
  bool isValidBST(TreeNode* root) {
    return isValidBST_helper(root, LONG_MAX, LONG_MIN);
  }

  bool isValidBST1(TreeNode* root) {
    TreeNode* cur = root, *pre = nullptr;
    stack<TreeNode*> st{};
    while (!st.empty() || nullptr != cur) {
      while (nullptr != cur) {
        st.emplace(cur);
        cur = cur->left;
      }
      cur = st.top(), st.pop();
      if (nullptr != pre && pre->val >= cur->val) return false;
      pre = cur, cur = cur->right;
    }
    return true;
  }

  //https://leetcode.com/problems/kth-smallest-element-in-a-bst/
  int kthSmallest(TreeNode* root, int k) {
    TreeNode* cur = root;
    stack<TreeNode*> st{};
    while (!st.empty() || nullptr != cur) {
      while (nullptr != cur) {
        st.emplace(cur);
        cur = cur->left;
      }
      cur = st.top(), st.pop();
      k--;
      if (k == 0) return cur->val;
      cur = cur->right;
    }
    return -1;
  }

  //https://leetcode.com/problems/inorder-successor-in-bst/
  TreeNode* inorderSuccessor(TreeNode* root, TreeNode* p) {
    std::stack<TreeNode*> st{};
    TreeNode* cur = root;
    bool flag = false;
    while (!st.empty() || nullptr != cur) {
      while (nullptr != cur) {
        st.emplace(cur);
        cur = cur->left;
      }
      cur = st.top(), st.pop();
      if (flag) return cur;
      if (cur == p) flag = true;
      cur = cur->right;
    }
    return nullptr;
  }

  //逼近思想, 用左子树不断逼近上限
  TreeNode* inorderSuccessor1(TreeNode* root, TreeNode* p) {
    TreeNode* res = nullptr;
    while (nullptr != root) {
      if (root->val > p->val) {
        res = root;
        root = root->left;
      } else {
        root = root->right;
      }
    }
    return res;
  }
  //前驱实现和后继思路一致, 用右子树不断逼近下限
  TreeNode* inorderPredecessor(TreeNode* root, TreeNode* p) {
    TreeNode* res = nullptr;
    while (nullptr != root) {
      if (root->val < p->val) {
        res = root;
        root = root->right;
      } else {
        root = root->left;
      }
    }
    return res;
  }

  //https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/
  TreeNode* treeToDoublyList(TreeNode* root) {
    std::stack<TreeNode*> st{};
    TreeNode *head, *cur, *pre;
    cur = root;
    head = pre = nullptr;
    while (!st.empty() || nullptr != cur) {
      while (nullptr != cur) {
        st.emplace(cur);
        cur = cur->left;
      }
      cur = st.top(), st.pop();
      if (nullptr == head) head = cur;
      if (nullptr != pre) {
        pre->right = cur;
        cur->left = pre;
      }
      pre = cur;
      cur = cur->right;
    }
    pre->right = head;
    head->left = pre;
    return head;
  }

  //https://leetcode.com/problems/minimum-distance-between-bst-nodes/
  int minDiffInBST(TreeNode* root) {
    std::stack<TreeNode*> st{};
    TreeNode *cur = root;
    int res = INT_MAX, pre_val = INT_MAX;
    while (!st.empty() || nullptr != cur) {
      while (nullptr != cur) {
        st.emplace(cur);
        cur = cur->left;
      }
      cur = st.top(), st.pop();
      if (cur->val > pre_val && cur->val - pre_val < res) {
        res = cur->val - pre_val;
      }
      pre_val = cur->val;
      cur = cur->right;
    }
    return res;
  }

  //https://leetcode.com/problems/closest-binary-search-tree-value/
  int closestValue(TreeNode* root, double target) {
    double min_diff = numeric_limits<double>::max();
    int res = root->val;
    while (nullptr != root) {
      double diff = std::abs(root->val - target);
      if (diff < min_diff) {
        res = root->val;
        min_diff = diff;
      }
      root = root->val > target ? root->left : root->right;
    }
    return res;
  }
  //https://leetcode.com/problems/closest-binary-search-tree-value-ii/
  void inorder(TreeNode *root, double target, int k, vector<int> &res) {
    if (nullptr == root) return ;
    inorder(root->left, target, k, res);

    if (res.size() < k) {
      res.emplace_back(root->val);
    } else if (abs(root->val - target) < abs(res[0] - target)) {
      res.erase(res.begin());
      res.emplace_back(root->val);
    } else {
      return ;
    }

    inorder(root->right, target, k, res);
  }
  //iteration
  vector<int> closestKValues(TreeNode* root, double target, int k) {
    vector<int> res{};
    inorder(root, target, k, res);
    return res;
  }

  //non iteration
  vector<int> closestKValues1(TreeNode* root, double target, int k) {
    vector<int> res{};
    TreeNode* cur = root;
    stack<TreeNode*> sta{};
    while (root != nullptr || !sta.empty()) {
      while (root != nullptr) {
        sta.emplace(root);
        cur = cur->left;
      }

      cur = sta.top(); sta.pop();
      if (k > res.size()) {
        res.emplace_back(cur->val);
      } else if (abs(cur->val - target) < abs(res[0] - target)) {
        res.erase(res.begin());
        res.emplace_back(cur->val);
      } else {
        break;
      }

      cur = cur->right;
    }
    return res;
  }

  //priority queue
  vector<int> closestKValues2(TreeNode* root, double target, int k) {
    vector<int> res{};
    priority_queue<pair<double, int>> que{};
    inorder(root, target, k, que);

    while (!que.empty()) {
      res.emplace_back(que.top().second);
      que.pop();
    }
    return res;
  }
  void inorder(TreeNode* root, double target, int k, priority_queue<pair<double, int>>& que) {
    if (root == nullptr) return ;
    inorder(root->left, target, k, que);
    que.emplace(abs(target - root->val), root->val);
    if (que.size() > k) que.pop();
    inorder(root->right, target, k, que);
  }

  //https://leetcode.com/problems/generate-parentheses/
  vector<string> generateParenthesis(int n) {
    vector<string> res {};
    backtrack(res, "", n, 0);
    return res;
  }
  void backtrack(vector<string> &res, const string& s, int l, int r) {
    if (l == 0 && r == 0) {
      res.push_back(s);
      return;
    }
    if (l > 0) backtrack(res, s + "(", l-1, r+1);
    if (r > 0) backtrack(res, s + ")", l, r-1);
  }

  vector<string> generateParenthesis1(int n) {
    vector<string> res{};
    helper(res, "", n, n);
    return res;
  }
  void helper(vector<string>& res, const string& str, int l, int r) {
    if (l == 0 && r == 0) {
      res.emplace_back(str);
      return ;
    }
    if (l > 0) helper(res, str + "(", l-1, r);
    if (r > l) helper(res, str + ")", l, r-1);
  }

  //stack
  vector<string> generateParenthesis2(int n) {
    stack<tuple<string, int, int>> st{};
    vector<string> res{};
    st.emplace("", 0, 0);
    while (!st.empty()) {
      tuple<string, int, int> tup = st.top(); st.pop();
      int l = std::get<1>(tup), r = std::get<2>(tup);
      if (l == n && r == n) {
        res.emplace_back(std::get<0>(tup));
      }
      if (l < n) st.emplace(std::get<0>(tup) + "(", l+1, r);
      if (l > r) st.emplace(std::get<0>(tup) + ")", l, r+1);
    }
    return res;
  }

  //https://leetcode.com/problems/top-k-frequent-elements/
  vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> m{};
    for (auto n : nums) {
      m[n] += 1;
    }

    vector<vector<int>> dp(nums.size()+1, vector<int>());
    for (auto& item : m) {
      dp[item.second].emplace_back(item.first);
    }

    vector<int> res{};
    for (int i = dp.size() - 1; i > 0; i--) {
      if (!dp[i].empty() && k >= dp[i].size()) {
        res.insert(res.end(), dp[i].begin(), dp[i].end());
        k -= dp[i].size();
      }
    }
    return res;
  }

  //use priority_queue
  vector<int> topKFrequent1(vector<int>& nums, int k) {
    unordered_map<int, int> m{};
    for (auto n : nums) {
      m[n] += 1;
    }

    vector<int> res{};
    priority_queue<pair<int, int>> pq{};
    for (auto& item : m) {
      pq.emplace(item.second, item.first);
      if (pq.size() > m.size() - k) {
        res.emplace_back(pq.top().second);
        pq.pop();
      }
    }
    return res;
  }

  //https://leetcode.com/problems/product-of-array-except-self/
  vector<int> productExceptSelf(vector<int>& nums) {
    int len = nums.size();
    vector<int> res(len, 1);
    for (int i = 1; i < len; i++) {
      res[i] = res[i-1] * nums[i-1];
    }

    int right = 1;
    for (int i = len - 1; i >= 0; i--) {
      res[i] *= right;
      right *= nums[i];
    }

    return res;
  }

  //https://leetcode.com/problems/group-anagrams/
  vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> mid{};
    for (auto& str : strs) {
      string s1 = str;
      std::sort(s1.begin(), s1.end());
      mid[s1].emplace_back(str);
    }

    vector<vector<string>> res{};
    for (auto& item : mid) {
      res.emplace_back(item.second);
    }
    return res;
  }

  string bucket_str(const string& str) {
    vector<int> bucket(26, 0);
    for (auto& ch : str) {
      bucket[ch - 'a']++;
    }

    string res = "";
    for (int i = 0; i < bucket.size(); i++) {
      if (bucket[i]) {
        res += std::to_string(bucket[i]);
        res += char('a' + i);
      }
    }
    return res;
  }
  vector<vector<string>> groupAnagrams1(vector<string>& strs) {
    unordered_map<string, vector<string>> mid{};
    for (auto& str : strs) {
      mid[bucket_str(str)].emplace_back(str);
    }

    vector<vector<string>> res{};
    for (auto& item : mid) {
      res.emplace_back(item.second);
    }
    return res;
  }

  //https://leetcode.com/problems/rotate-image/
  void rotate(vector<vector<int>>& matrix) {
    int l = 0, r = matrix.size()-1;
    cout << r << endl;
    while (l < r) {
      std::swap(matrix[l], matrix[r]);
      l++;r--;
    }

    for (int i = 0; i < matrix.size(); i++) {
      for (int j = i+1; j < matrix.size(); j++) {
        int tmp = matrix[i][j];
        matrix[i][j] = matrix[j][i];
        matrix[j][i] = tmp;
      }
    }
  }

  //https://leetcode.com/problems/find-words-that-can-be-formed-by-characters/
  int countCharacters(vector<string>& words, string chars) {
    unordered_map<char, int> dp{};
    for (char& ch : chars) {
      dp[ch]++;
    }

    int res = 0;
    for (string& word : words) {
      unordered_map<char, int> tmp = dp;
      bool is_match = true;
      for (char& ch : word) {
        if (tmp[ch] <= 0) {
          is_match = false;
          break;
        }
        tmp[ch]--;
      }

      if (is_match) res += word.size();
    }
    return res;
  }
  int countCharacters1(vector<string>& words, string chars) {
    unordered_map<char, int> dp{};
    for (char& ch : chars) dp[ch]++;

    int res = 0;
    bool is_match;
    for (auto& word : words) {
      is_match = true;
      unordered_map<char, int> tmp{};
      for (auto& ch : word) tmp[ch]++;
      for (auto& ch : word) {
        if (dp[ch] < tmp[ch]) {
          is_match = false;
        }
      }
      if (is_match) res += word.size();
    }
    return res;
  }

  //https://leetcode.com/problems/odd-even-linked-list/
  ListNode *oddEvenList(ListNode *head) {
    auto *even = new ListNode(0);
    ListNode *even_crt = even;
    auto *odd = new ListNode(0);
    ListNode *odd_crt = odd;

    int n = 1;
    while (head) {
      if ((n & 1) == 1) {
        odd_crt->next = head;
        odd_crt = odd_crt->next;
      } else {
        even_crt->next = head;
        even_crt = even_crt->next;
      }
      head = head->next;
      n++;
    }
    odd_crt->next = even->next;
    even_crt->next = NULL;
    return odd->next;
  }

  ListNode *oddEvenList1(ListNode *head) {
    if (head == nullptr) return head;
    ListNode* odd = head, *even = head->next, *even_head = even;
    while (even != nullptr && even->next != nullptr) {
      odd->next = even->next;
      even->next = even->next->next;
      odd = odd->next;
      even = even->next;
    }
    odd->next = even_head;
    return head;
  }

  //https://leetcode.com/problems/find-the-duplicate-number/
  int findDuplicate(vector<int>& nums) {
    if (nums.empty()) return -1;

    int low = 1, high = nums.size() - 1;
    while (low < high) {
      int mid = (low + high) >> 1;
      int count = 0;
      for (int i = 0; i < nums.size(); i++) {
        if (nums[i] <= mid) {
          count++;
        }
      }

      if (count > mid) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    return low;
  }

  int findDuplicate1(vector<int>& nums) {
    if (nums.empty()) return -1;
    int fast = nums[nums[0]], slow = nums[0];
    while (fast != slow) {
      slow = nums[slow];
      fast = nums[nums[fast]];
    }

    slow = 0;
    while (fast != slow) {
      fast = nums[fast];
      slow = nums[slow];
    }
    return slow;
  }

  //https://leetcode.com/problems/set-mismatch/
  vector<int> findErrorNums(vector<int>& nums) {
    vector<int> vec(nums.size(), 0);
    vector<int> res(2, 0);
    for (int i = 0; i < nums.size(); i++) {
      vec[nums[i] - 1]++;
      res[1] += (i + 1 - nums[i]);
      if (vec[nums[i] - 1] >= 2) {
        res[0] = nums[i];
      }
    }
    res[1] += res[0];
    return res;
  }

  vector<int> findErrorNums1(vector<int>& nums) {
    vector<int> res(2, 0);
    for (int i = 0; i < nums.size(); i++) {
      int idx = abs(nums[i]) - 1;
      res[1] += (i - idx);
      if (nums[idx] < 0) {
        res[0] = idx + 1;
      } else {
        nums[idx] = -nums[idx];
      }
    }
    res[1] += res[0];
    return res;
  }

  //https://leetcode.com/problems/linked-list-cycle-ii/
  ListNode *detectCycle(ListNode *head) {
    ListNode *slow = head, *fast = head;
    while (fast != nullptr && fast->next != nullptr) {
      slow = slow->next;
      fast = fast->next->next;
      if (slow == fast) break;
    }
    if (fast == nullptr || fast->next == nullptr) return nullptr;

    slow = head;
    while (slow != fast) {
      slow = slow->next;
      fast = fast->next;
    }
    return fast;
  }

  //https://leetcode.com/problems/first-missing-positive/
  int firstMissingPositive(vector<int>& nums) {
    //put num on it's right location
    int n = nums.size();
    for (int i = 0; i < n; i++) {
      while (nums[i] > 0 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
        std::swap(nums[i], nums[nums[i] - 1]);
      }
    }

    for (int i = 0; i < n; i++) {
      if (nums[i] != i + 1) {
        return i + 1;
      }
    }
    return n + 1;
  }

  //https://leetcode.com/problems/kth-largest-element-in-an-array/
  int partition(vector<int>& nums, int left, int right) {
    int key = nums[left];
    while (left < right) {
      while (left < right && nums[right] < key) right--;
      nums[left] = nums[right];
      while (left < right && nums[left] >= key) left++;
      nums[right] = nums[left];
    }
    nums[left] = key;
    return left;
  }
  int findKthLargest1(vector<int>& nums, int k) {
    int left = 0, right = nums.size() - 1, pos;
    k--;
    while (left < right) {
      pos = partition(nums, left, right);
      if (pos == k) {
        break;
      } else if (pos > k) {
        right = pos - 1;
      } else {
        left = pos + 1;
      }
    }
    return nums[k];
  }

  int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> pq{};
    for (auto& num : nums) {
      pq.emplace(num);
      if (pq.size() > k) {
        pq.pop();
      }
    }
    return pq.top();
  }

  //https://leetcode.com/problems/wiggle-sort-ii/
//  void wiggleSort(vector<int>& nums) {
//    std::sort(nums.begin(), nums.end());
//    vector<bool> is_used(nums.size(), false);
//    int begin = 0, end = nums.size()-1;
//    for ()
//  }


  //https://leetcode.com/problems/third-maximum-number/
  int thirdMax1(vector<int>& nums) {
    priority_queue<int, vector<int>, greater<int>> pq{};
    unordered_set<int> num_set{};
    int max_v = INT_MIN;
    for (auto& num : nums) {
      if (num_set.find(num) == num_set.end()) {
        max_v = std::max(max_v, num);
        pq.emplace(num);
        num_set.emplace(num);
        if (pq.size() > 3) pq.pop();
      }
    }
    return num_set.size() >= 3 ? pq.top() : max_v;
  }

  // can't ac, just an idea
  int thirdMax(vector<int>& nums) {
    int max1, max2, max3;
    max1 = max2 = max3 = INT_MIN;
    for (auto& num : nums) {
      if (num == max1 || num == max2 || num == max3) continue;
      if (num > max1) {
        max3 = max2;
        max2 = max1;
        max1 = num;
      } else if (num > max2) {
        max3 = max2;
        max2 = num;
      } else if (num > max3) {
        max3 = num;
      }
    }
    return max3 > INT_MIN ? max3 : max1;
  }

  //https://leetcode.com/problems/container-with-most-water/
  int maxArea(vector<int>& nums) {
    int res = 0, len = nums.size() - 1;
    for (int i = 0; i < len; i++) {
      for (int j = len; j > i; j--) {
        res = std::max(res, (j - i) * std::min(nums[i], nums[j]));
      }
    }
    return res;
  }

  int maxArea1(vector<int>& nums) {
    int res = 0, len = nums.size() - 1;
    for (int i = 0; i < len; i++) {
      if (i > 0 && nums[i] <= nums[i-1]) continue;
      for (int j = len; j > i; j--) {
        if (nums[j] >= nums[i]) {
          res = std::max(res, (j - i) * nums[i]);
          break;
        }
        if (j < len && nums[j] <= nums[j+1]) continue;
        res = std::max(res, (j - i) * std::min(nums[i], nums[j]));
      }
    }
    return res;
  }

  int maxArea2(vector<int>& nums) {
    int res = 0, max_l = 0, max_r = 0, end = nums.size() - 1;
    for (int i = 0; i < end; i++) {
      if (nums[i] <= max_l) continue; //obsolete right
      max_l = nums[i];
      max_r = 0;

      for (int j = end; j > i; j--) {
        if (nums[j] >= nums[i]) {
          res = std::max(res, (j - i) * nums[i]);
          break;
        }

        end = j - 1; //obsolete left

        if (nums[j] <= max_r) continue;
        max_r = nums[j];
        res = std::max(res, (j - i) * std::min(nums[i], nums[j]));
      }
    }
    return res;
  }

  int maxArea3(vector<int>& nums) {
    int start = 0, end = nums.size() - 1, res = 0;
    while (start < end) {
      res = std::max(res, std::min(nums[start], nums[end]) * (end - start));
      if (nums[start] < nums[end]) {
        start++;
      } else {
        end--;
      }
    }
    return res;
  }

  //https://leetcode.com/problems/k-closest-points-to-origin/
  vector<vector<int>> kClosest1(vector<vector<int>>& points, int K) {
    priority_queue<pair<int, vector<int>>> pq{};
    for (auto& p : points) {
      pq.emplace(make_pair(p[0]*p[0] + p[1]*p[1], p));
      if (pq.size() > K) pq.pop();
    }

    vector<vector<int>> res{};
    while (!pq.empty()) {
      res.emplace_back(pq.top().second);
      pq.pop();
    }
    return res;
  }

  vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
    priority_queue<pair<int, vector<int>>, vector<pair<int, vector<int>>>, greater<pair<int, vector<int>>>> pq{};
    int prio = points.size() - K;
    vector<vector<int>> res{};
    for (auto& p : points) {
      pq.emplace(make_pair(p[0]*p[0] + p[1]*p[1], std::move(p)));
      if (pq.size() > prio) {
        res.emplace_back(pq.top().second);
        pq.pop();
      }
    }
    return res;
  }

  //quick sort
  int position(vector<int>& nums, int begin, int end) {
    int key = nums[begin];
    while (begin < end) {
      while (begin < end && nums[end] >= key) end--;
      nums[begin] = nums[end];
      while (begin < end && nums[begin] < key) begin++;
      nums[end] = nums[begin];
    }
    nums[begin] = key;
    return begin;
  }

  void quick_sort(vector<int>& nums, int begin, int end) {
    if (begin < end) {
      int mid = position(nums, begin, end);
      quick_sort(nums, begin, mid - 1);
      quick_sort(nums, mid + 1, end);
    }
  }
  //heap sort, big heap
  void adjust(vector<int>& nums, int root, int len) {
    int child = root * 2 + 1;
    if (child < len && (child + 1) < len && nums[child] < nums[child+1]) {
      child++;
    }

    if (child < len && nums[root] < nums[child]) {
      std::swap(nums[root], nums[child]);
      adjust(nums, child, len);
    }
  }

  void heap_sort(vector<int>& nums) {
    int len = nums.size();
    //build heap
    for (int i = (len / 2 - 1); i >= 0; i--) {
      adjust(nums, i, len);
    }
    //adjust
    for (int i = len - 1; i > 0; i--) {
      std::swap(nums[0], nums[i]);
      adjust(nums, 0, i);
    }
  }

  //kth large num
  int kth_large(vector<int>& nums, int k) {
    int begin = 0, end = nums.size() - 1;
    k--;
    while (begin <= end) {
      int mid = position(nums, begin, end);
      if (mid == k) {
        return nums[mid];
      } else if (mid > k) {
        end = mid - 1;
      } else {
        begin = mid + 1;
      }
    }
    return -1;
  }
  //bucket sort

  //https://leetcode.com/problems/game-of-life/
  void gameOfLife(vector<vector<int>>& board) {
    int row = board.size(), col = board[0].size();
    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        int lives = 0;
        for (int m = i - 1; m <= i + 1; m++) {
          for (int n = j - 1; n <= j + 1; n++) {
            if ((m >= 0 && m < row) && (n >= 0 && n < col) && (board[m][n] == 1 || board[m][n] == 2)) {
              lives++;
            }
          }
        }

        if (board[i][j] == 1 && (lives < 3 || lives > 4)) { //add board[i][j]=1 more
          board[i][j] = 2;
        } else if (board[i][j] == 0 && lives == 3) {
          board[i][j] = 3;
        }
      }
    }

    for (auto& vec : board) {
      for (auto& num : vec) {
        num %= 2;
      }
    }
  }

  //https://leetcode.com/problems/spiral-matrix/
  vector<int> spiralOrder(vector<vector<int>>& mat) {
    vector<int> res{};
    if (mat.empty()) return res;

    int r1 = 0, r2 = mat.size()-1, c1 = 0, c2 = mat[0].size()-1;
    int i = r1, j = c1;
    while (r1 <= r2 && c1 <= c2) {
      if (i == r1 && j == c1) {
        while (j <= c2) res.emplace_back(mat[i][j++]);
        j = c2;
        r1++;
        i++;
      } else if (i == r1 && j == c2) {
        while (i <= r2) res.emplace_back(mat[i++][j]);
        i = r2;
        c2--;
        j--;
      } else if (i == r2 && j == c2) {
        while (j >= c1) res.emplace_back(mat[i][j--]);
        j = c1;
        r2--;
        i--;
      } else {
        while (i >= r1) res.emplace_back(mat[i--][j]);
        i = r1;
        c1++;
        j++;
      }
    }
    return res;
  }

  //https://leetcode.com/problems/spiral-matrix-ii/
  vector<vector<int>> generateMatrix(int n) {
    vector<vector<int>> res(n, vector<int>(n, 0));
    int r1 = 0, r2 = n - 1, c1 = 0, c2 = n - 1;
    int i = r1, j = c1, num = 1;
    while (r1 <= r2 && c1 <= c2) {
      if (i == r1 && j == c1) {
        while (j <= c2) res[i][j++] = num++;
        j = c2;
        r1++;
        i++;
      } else if (i == r1 && j == c2) {
        while (i <= r2) res[i++][j] = num++;
        i = r2;
        c2--;
        j--;
      } else if (i == r2 && j == c2) {
        while (j >= c1) res[i][j--] = num++;
        j = c1;
        r2--;
        i--;
      } else {
        while (i >= r1) res[i--][j] = num++;
        i = r1;
        c1++;
        j++;
      }
    }
    return res;
  }

  //https://leetcode.com/problems/binary-tree-level-order-traversal/
  vector<vector<int>> levelOrder1(TreeNode* root) {
    vector<vector<int>> res{};
    if (root == nullptr) return res;

    std::queue<TreeNode*> q{};
    q.emplace(root);
    while (!q.empty()) {
      int level_size = q.size();
      vector<int> level_res(level_size);
      for (int i = 0; i < level_size; i++) {
        root = q.front();
        level_res[i] = root->val;
        if (root->left != nullptr) q.emplace(root->left);
        if (root->right != nullptr) q.emplace(root->right);
        q.pop();
      }
      res.emplace_back(level_res);
    }
    return res;
  }

  void levelOrderHelper(TreeNode *root, int deep, vector<vector<int>>& res) {
    if (root != nullptr) {
      if (res.size() > deep) {
        res[deep].emplace_back(root->val);
      } else {
        res.emplace_back(vector<int>{root->val});
      }
      levelOrderHelper(root->left, deep+1, res);
      levelOrderHelper(root->right, deep+1, res);
    }
  }
  vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> res{};
    levelOrderHelper(root, 0, res);
    return res;
  }

  //https://leetcode.com/problems/binary-tree-level-order-traversal-ii/
  vector<vector<int>> levelOrderBottom(TreeNode* root) {
    vector<vector<int>> res{};
    levelOrderHelper(root, 0, res);

    int left = 0, right = res.size() - 1;
    while (left < right) {
      std::swap(res[left], res[right]);
      left++;
      right--;
    }
    return res;
  }

  //https://leetcode.com/problems/minimum-depth-of-binary-tree/
  int minDepth(TreeNode* root) {
    if (root == nullptr) {
      return 0;
    }
    if (root->left == nullptr || root->right == nullptr) {
      return 1 + minDepth(root->right) + minDepth(root->left);
    }
    return 1 + std::min(minDepth(root->left), minDepth(root->right));
  }

  //https://leetcode.com/problems/average-of-levels-in-binary-tree/
  void avgLevelsHelper(TreeNode* root, vector<pair<int, long>>& res, int deep) {
    if (root != nullptr) {
      if (res.size() > deep) {
        auto& item = res[deep];
        item.first += 1;
        item.second += root->val;
      } else {
        res.emplace_back(1, root->val);
      }

      avgLevelsHelper(root->left, res, deep + 1);
      avgLevelsHelper(root->right, res, deep + 1);
    }
  }

  //use int can't ac, value too large to sum
  vector<double> averageOfLevels(TreeNode* root) {
    vector<pair<int, long>> sum{};
    avgLevelsHelper(root, sum, 0);
    vector<double> res{};
    for (auto& item : sum) {
      res.emplace_back(item.second * 1.0 / item.first);
    }
    return res;
  }

  //https://leetcode.com/problems/cousins-in-binary-tree/
  bool isCousins(TreeNode* root, int x, int y) {
    if (root == nullptr) return false;
    std::queue<TreeNode*> que{};
    que.emplace(root);
    while (!que.empty()) {
      int l_size = que.size(), flag = 0;
      for (int i = 0; i < l_size; i++) {
        root = que.front();
        que.pop();
        if (root->val == x) flag |= 1;
        if (root->val == y) flag |= 2;

        if (root->left != nullptr && root->right != nullptr) {//not sister
          if ((root->left->val + root->right->val) == (x + y) && (root->left->val * root->right->val == x * y)) {
            return false;
          }
        }

        if (root->left != nullptr) que.emplace(root->left);
        if (root->right != nullptr) que.emplace(root->right);
      }

      if (flag == 3) return true;
      if (flag > 0) return false;
    }
    return false;
  }

  //https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/
  void zigzagLevelOrderHelper(TreeNode* root, vector<vector<int>>& res, int deep) {
    if (root != nullptr) {
      if (res.size() > deep) {
        if (!(deep & 1)) {
          res[deep].emplace_back(root->val);
        } else {
          res[deep].insert(res[deep].begin(), root->val);
        }
      } else {
        res.emplace_back(vector<int>{root->val});
      }

      zigzagLevelOrderHelper(root->left, res, deep + 1);
      zigzagLevelOrderHelper(root->right, res, deep + 1);
    }
  }

  vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
    vector<vector<int>> res{};
    zigzagLevelOrderHelper(root, res, 0);
    return res;
  }

  //https://leetcode.com/problems/binary-tree-vertical-order-traversal/
  vector<vector<int>> verticalOrder(TreeNode* root) {
    vector<vector<int>> res{};
    if (!root) return res;

    std::map<int, vector<int>> m{};
    std::queue<pair<int, TreeNode*>> que;
    que.emplace(0, root);
    while (!que.empty()) {
      auto& node = que.front();
      que.pop();
      m[node.first].emplace_back(node.second->val);

      if (node.second->left) que.emplace(node.first - 1, node.second->left);
      if (node.second->right) que.emplace(node.first + 1, node.second->right);
    }

    for (auto& item : m) {
      res.emplace_back(item.second);
    }
    return res;
  }

  //https://leetcode.com/problems/n-ary-tree-level-order-traversal/
  void levelOrderHelper(Node* root, vector<vector<int>>& res, int deep) {
    if (root != nullptr) {
      if (res.size() > deep) {
        res[deep].emplace_back(root->val);
      } else {
        res.emplace_back(vector<int>{root->val});
      }

      for (auto& child : root->children) {
        levelOrderHelper(child, res, deep + 1);
      }
    }
  }

  vector<vector<int>> levelOrder(Node* root) {
    vector<vector<int>> res{};
    levelOrderHelper(root, res, 0);
    return res;
  }

  //https://leetcode.com/problems/n-ary-tree-preorder-traversal/
  void preOrderHelper(Node* root, vector<int>& res) {
    if (!root) return;
    res.emplace_back(root->val);
    for (auto& child : root->children) {
      preOrderHelper(child, res);
    }
  }
  vector<int> preorder1(Node* root) {
    vector<int> res{};
    preOrderHelper(root, res);
    return res;
  }

  vector<int> preorder(Node* root) {
    vector<int> res{};
    if (!root) return res;

    stack<Node*> sta{};
    sta.emplace(root);
    while (!sta.empty()) {
      Node* cur = sta.top();
      sta.pop();
      res.emplace_back(cur->val);
      for (int i = cur->children.size() - 1; i >= 0; i--) {
        sta.emplace(cur->children[i]);
      }
    }
    return res;
  }

  //https://leetcode.com/problems/n-ary-tree-postorder-traversal/
  vector<int> postorder(Node* root) {
    vector<int> res{};
    if (!root) return res;

    stack<Node*> st{};
    st.emplace(root);
    while (!st.empty()) {
      Node* cur = st.top();
      st.pop();
      res.insert(res.begin(), cur->val);
      for (int i = 0; i < cur->children.size(); i++) {
        st.emplace(cur->children[i]);
      }
    }
    return res;
  }

  //https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
  void heapBuild(vector<int>& data, int root, int len) {
    int left = root * 2 + 1;
    if (left < len) {
      if ((left + 1) < len && data[left] < data[left + 1]) {
        left++;
      }
      if (data[left] > data[root]) {
        std::swap(data[left], data[root]);
        heapBuild(data, left, len);
      }
    }
  }

  int kthSmallest(vector<vector<int>>& matrix, int k) {
    vector<int> maxHeap(k, INT_MAX);
    int len = matrix.size();
    for (int i = 0; i < len; i++) {
      if (matrix[i][0] > maxHeap[0]) break;
      for (int j = 0; j < len; j++) {
        if (matrix[i][j] > maxHeap[0]) {
          break;
        } else {
          maxHeap[0] = matrix[i][j];
          heapBuild(maxHeap, 0, k);
        }
      }
    }
    return maxHeap[0];
  }

  int kthSmallest1(vector<vector<int>>& matrix, int k) {
    priority_queue<int> pq{};
    for (int i = 0; i < matrix.size(); i++) {
      for (int j = 0; j < matrix[0].size(); j++) {
        if (pq.size() < k) {
          pq.emplace(matrix[i][j]);
        } else {
          if (matrix[i][j] >= pq.top()) {
            break;
          }
          pq.emplace(matrix[i][j]);
          pq.pop();
        }
      }
    }
    return pq.top();
  }

  int kthSmallest2(vector<vector<int>>& matrix, int k) {
    unsigned int n = matrix.size();
    int l = matrix[0][0], r = matrix[n-1][n-1], count, mid;
    while (l < r) {
      mid = (l + r) >> 1;
      count = 0;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n && matrix[i][j] <= mid; j++) {
          count++;
        }
      }

      if (count < k) {
        l = mid + 1;
      } else {
        r = mid;
      }
    }
    return l;
  }

  //https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/
  vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
    vector<vector<int>> res{};
    if (nums1.empty() || nums2.empty() || k <= 0) return res;

    int e1 = nums1.size(), e2 = nums2.size();
    priority_queue<pair<int, pair<int, int>>> pq{};
    for (int i = 0; i < e1 && i < k; i++) {
      pq.emplace(0 - nums1[i] - nums2[0], make_pair(i, 0));
    }

    while (k > 0 && !pq.empty()) {
      auto item = pq.top(); pq.pop();
      int idx1 = item.second.first, idx2 = item.second.second;
      res.emplace_back(vector<int>{nums1[idx1], nums2[idx2]});
      if (idx2 + 1 < e2) {
        pq.emplace(0 - nums1[idx1] - nums2[idx2 + 1], make_pair(idx1, idx2 + 1));
      }
      k--;
    }
    return res;
  }

  //https://leetcode.com/problems/unique-paths/
  int uniquePaths1(int m, int n) {
    vector<vector<int>> dp(m, vector<int>(n, 1));
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        dp[i][j] = dp[i-1][j] + dp[i][j-1];
      }
    }
    return dp[m-1][n-1];
  }

  int uniquePaths(int m, int n) {
    vector<int> dp(n, 1);
    for (int i = 1; i < m; i++) {
      for (int j = 1; j < n; j++) {
        dp[j] += dp[j-1];
      }
    }
    return dp[n-1];
  }

  //https://leetcode.com/problems/unique-paths-ii/
  int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    if (obstacleGrid.empty() || obstacleGrid[0].empty()) return 0;
    int row = obstacleGrid.size(), col = obstacleGrid[0].size();
    vector<int> dp(col, 0);
    dp[0] = 1;

    for (int i = 0; i < row; i++) {
      for (int j = 0; j < col; j++) {
        if (obstacleGrid[i][j] == 1) {
          dp[j] = 0;
        } else if (j > 0) {
          dp[j] += dp[j-1];
        }
      }
    }

    return dp[col-1];
  }

  //https://leetcode.com/problems/array-nesting/
  int arrayNesting1(vector<int>& nums) {
    unordered_set<int> s{};
    int count, index, res = 0;
    for (int i = 0; i < nums.size(); i++) {
      index = i;
      count = 0;
      while (s.find(index) == s.end()) {
        s.emplace(index);
        index = nums[index];
        count++;
      }
      res = std::max(res, count);
    }
    return res;
  }

  int arrayNesting(vector<int>& nums) {
    int count, index, res = 0;
    for (int i = 0; i < nums.size(); i++) {
      index = i;
      count = 0;
      while (index >= 0 && nums[index] >= 0) {
        int next = nums[index];
        nums[index] = -1; // nums[visited] = -1
        index = next;
        count++;
      }
      res = std::max(res, count);
    }
    return res;
  }

  //https://leetcode.com/problems/nested-list-weight-sum/
  //dfs
  int depthSumHelper(const vector<NestedInteger>& n_list, int depth) {
    int sum = 0;
    for (auto& item : n_list) {
      if (item.isInteger()) {
        sum += (depth * item.getInteger());
      } else {
        sum += depthSumHelper(item.getList(), depth + 1);
      }
    }
    return sum;
  }

  int depthSum1(vector<NestedInteger>& nestedList) {
    return depthSumHelper(nestedList, 1);
  }

  int depthSum(vector<NestedInteger>& nestedList) {
    stack<pair<NestedInteger, int>> st{};
    for (auto& item : nestedList) {
      st.emplace(item, 1);
    }

    int sum = 0;
    while (!st.empty()) {
      auto& item = st.top();
      st.pop();
      if (item.first.isInteger()) {
        sum += (item.first.getInteger() * item.second);
      } else {
        for (auto& item1 : item.first.getList()) {
          st.emplace(item1, item.second + 1);
        }
      }
    }
    return sum;
  }

  //https://leetcode.com/problems/nested-list-weight-sum-ii/
  //bfs
  int depthSumInverse(vector<NestedInteger>& nestedList) {
    queue<NestedInteger> q{};
    int res = 0, mid = 0;
    while (!nestedList.empty()) {
      vector<NestedInteger> next{};
      for (auto& item : nestedList) {
        if (item.isInteger()) {
          mid += item.getInteger();
        } else {
          next.insert(next.begin(), item.getList().begin(), item.getList().end());
        }
      }
      res += mid;
      nestedList = next;
    }
    return res;
  }

  //https://leetcode.com/problems/shuffle-an-array/

  //https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
  TreeNode* buildTreeHelper(vector<int>& preorder, int pre_s,
    vector<int>& inorder, int in_s, int in_e, unordered_map<int, int>& idx_map) {
    if (pre_s >= preorder.size() || in_s > in_e) return nullptr;
    auto* root = new TreeNode(preorder[pre_s]);
    int idx = idx_map[preorder[pre_s]];
    root->left = buildTreeHelper(preorder, pre_s + 1, inorder, in_s, idx - 1, idx_map);
    root->right = buildTreeHelper(preorder, pre_s + idx - in_s + 1, inorder, idx + 1, in_e, idx_map);
    return root;
  }

  TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
    unordered_map<int, int> m{};
    for (int i = 0; i < inorder.size(); i++) {
      m[inorder[i]] = i;
    }
    return buildTreeHelper(preorder, 0, inorder, 0, inorder.size()-1, m);
  }

  //https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/
  TreeNode* postHelper(vector<int>& inorder, vector<int>& postorder,
    int in_s, int in_e, int post_s, int post_e, unordered_map<int, int>& in_map) {
    if (in_s > in_e || post_s > post_e) return nullptr;
    auto* root = new TreeNode(postorder[post_e]);
    int idx = in_map[postorder[post_e]];
    root->left = postHelper(inorder, postorder, in_s, idx - 1, post_s, post_s + idx - in_s - 1, in_map);
    root->right = postHelper(inorder, postorder, idx + 1, in_e, post_s + idx - in_s, post_e - 1, in_map);
    return root;
  }

  TreeNode* buildTree1(vector<int>& inorder, vector<int>& postorder) {
    unordered_map<int, int> m{};
    for (int i = 0; i < inorder.size(); i++) {
      m[inorder[i]] = i;
    }

    return postHelper(inorder, postorder, 0, inorder.size()-1, 0, postorder.size()-1, m);
  }

  //https://leetcode.com/problems/valid-sudoku/
  bool isValidSudoku1(vector<vector<char>>& board) {
    vector<int> prime{2,3,5,7,11,13,17,19,23};
    int multi = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23;
    vector<int> row(9, multi);
    vector<int> col(9, multi);
    vector<int> block(9, multi);
    for (int i = 0; i < board.size(); i++) {
      for (int j = 0; j < board[0].size(); j++) {
        if (board[i][j] != '.') {
          int idx = board[i][j] - '0' - 1;
          if (row[i] % prime[idx] != 0 || col[j] % prime[idx] != 0 || block[i / 3 * 3 + j / 3] % prime[idx] != 0) {
            return false;
          }
          row[i] /= prime[idx];
          col[j] /= prime[idx];
          block[i/3*3 + j/3] /= prime[idx];
        }
      }
    }
    return true;
  }

  bool isValidSudoku(vector<vector<char>>& board) {
    unordered_set<std::string> set{};
    for (int i = 0; i < board.size(); i++) {
      for (int j = 0; j < board[0].size(); j++) {
        if (board[i][j] != '.') {
          string row = std::to_string(i) + "row" + board[i][j];
          string col = std::to_string(j) + "col" + board[i][j];
          string block = std::to_string(i / 3 * 3 + j / 3) + "block" + board[i][j];
          if (set.find(row) != set.end() || set.find(col) != set.end() || set.find(block) != set.end()) {
            return false;
          }
          set.emplace(row);
          set.emplace(col);
          set.emplace(block);
        }
      }
    }
  }

  //https://leetcode.com/problems/sort-colors/
  void sortColors1(vector<int>& nums) {
    int s = 0, e = nums.size() - 1;
    for (int i = 0; i <= e; i++) {
      while (nums[i] == 2 && i <= e) std::swap(nums[i], nums[e--]);
      while (nums[i] == 0 && i >= s) std::swap(nums[i], nums[s++]);
    }
  }

  void sortColors(vector<int>& nums) {
    vector<int> rgb(3, 0);
    for (auto& n : nums) {
      rgb[n] += 1;
    }
    rgb[1] += rgb[0];
    rgb[2] += rgb[1];

    for (int i = 0; i < nums.size(); i++) {
      if (i < rgb[0]) {
        nums[i] = 0;
      } else if (i < rgb[1]) {
        nums[i] = 1;
      } else {
        nums[i] = 2;
      }
    }
  }

  //https://leetcode.com/problems/next-permutation/submissions/
  void nextPermutation(vector<int> &num) {
    //1　　2　　7　　4　　3　　1
    //1　　2　　7　　4　　3　　1
    //1　　3　　7　　4　　2　　1
    //1　　3　　1　　2　　4　　7
    int n = num.size() - 1;
    for (int i = n - 1; i >= 0; i--) {
      if (num[i] < num[i+1]) {
        for (int j = n; j > i; j--) {
          if (num[j] > num[i]) {
            std::swap(num[i], num[j]);
            std::reverse(num.begin() + i + 1, num.end());
            return;
          }
        }
      }
    }
    std::reverse(num.begin(), num.end());
  }

  //https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
  vector<int> searchRange(vector<int>& nums, int target) {
    vector<int> res(2, -1);
    int start = 0, end = nums.size() - 1;
    //left index
    while (start <= end) {
      int mid = (start + end) >> 1;
      if (nums[mid] < target) {
        start = mid + 1;
      } else {
        end = mid - 1;
      }
    }
    if (start >= nums.size() || nums[start] != target) return res;
    res[0] = start;

    //right index
    start = 0, end = nums.size() - 1;
    while (start <= end) {
      int mid = (start + end) >> 1;
      if (nums[mid] > target) {
        end = mid - 1;
      } else {
        start = mid + 1;
      }
    }
    res[1] = end;
    return res;
  }

  //https://leetcode.com/problems/first-bad-version/
  bool isBadVersion(int version);
  int firstBadVersion(int n) {
    int s = 1, mid, res = n;
    while (s <= n) {
      mid = s + (n - s) / 2;
      if (isBadVersion(mid)) {
        res = mid;
        n = mid - 1;
      } else {
        s = mid + 1;
      }
    }
    return res;
  }

  //https://leetcode.com/problems/search-insert-position/
  int searchInsert(vector<int>& nums, int target) {
    int s = 0, e = nums.size() - 1;
    while (s <= e) {
      int mid = s + (e - s) / 2;
      if (nums[mid] < target) {
        s = mid + 1;
      } else {
        e = mid - 1;
      }
    }
    return s;
  }

  //https://leetcode.com/problems/guess-number-higher-or-lower/
  int guess(int num);
  int guessNumber(int n) {
    int s = 1, e = n;
    while (s <= e) {
      int mid = s + (e - s) / 2;
      int pick = guess(mid);
      if (pick == 0) {
        return mid;
      } else if (pick == -1) {
        e = mid - 1;
      } else {
        s = mid + 1;
      }
    }
    return -1;
  }

  //https://leetcode.com/problems/sort-list/
  ListNode* merge(ListNode* l, ListNode* r) {
    auto *head = new ListNode(0);
    ListNode* cur = head;
    while (l != nullptr && r != nullptr) {
      if (r->val > l->val) {
        cur->next = l;
        l = l->next;
      } else {
        cur->next = r;
        r = r->next;
      }
      cur = cur->next;
    }
    if (l != nullptr) cur->next = l;
    if (r != nullptr) cur->next = r;
    return head->next;
  }

  ListNode* sortList(ListNode* head) {
    if (head == nullptr || head->next == nullptr) return head;
    //find mid
    ListNode *slow = head, *fast = head, *prev = head;
    while (fast != nullptr && fast->next != nullptr) {
      prev = slow;
      slow = slow->next;
      fast = fast->next->next;
    }
    prev->next = nullptr;

    //sort left & right
    ListNode* l = sortList(head);
    ListNode* r = sortList(slow);

    //merge
    return merge(l, r);
  }

  //https://leetcode.com/problems/insertion-sort-list/
  ListNode* insertionSortList(ListNode* head) {
    if (head == nullptr || head->next == nullptr) return head;

    ListNode node(0);
    ListNode *cur = head, *tail;
    while (cur != nullptr) {
      tail = &node;
      while (tail->next != nullptr && cur->val > tail->next->val) {
        tail = tail->next;
      }
      ListNode *tmp = tail->next;
      tail->next = cur;
      cur = cur->next;
      tail->next->next = tmp;
    }

    return node.next;
  }

  //https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/
  ListNode* insert(ListNode* head, int insertVal) {
    auto* res = new ListNode(insertVal);
    if (head == nullptr) {
      res->next = res;
      return res;
    }

    ListNode* cur = head;
    while (cur->next != head) {
      //插入中间 or 插入头尾
      if ((cur->val <= insertVal && cur->next->val >= insertVal) ||
      (cur->val > cur->next->val && (cur->val < insertVal || cur->next->val > insertVal))) {
        break;
      }
      cur = cur->next;
    }
    res->next = cur->next;
    cur->next = res;
    return head;
  }

  //https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
  TreeNode* lowestCommonAncestor1(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (root == nullptr || root == p || root == q) return root;
    TreeNode* left = lowestCommonAncestor(root->left, p, q);
    TreeNode* right = lowestCommonAncestor(root->right, p, q);
    if (left && right) return root;
    return left ? left : right;
  }

  //https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
  TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    if (root->val > p->val && root->val > q->val) {
      return lowestCommonAncestor(root->left, q, p);
    } else if (root->val < p->val && root->val < q->val) {
      return lowestCommonAncestor(root->right, p, q);
    }
    return root;
  }

  //https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-ii/
  ParentTreeNode* lowestCommonAncestorII(ParentTreeNode *root, ParentTreeNode *A, ParentTreeNode *B) {
    ParentTreeNode* cur_A = A, *cur_B = B;
    while (cur_A != cur_B) {
      if (cur_A != root) {
        cur_A = cur_A->parent;
      } else {
        cur_A = B;
      }

      if (cur_B != root) {
        cur_B = cur_B->parent;
      } else {
        cur_B = A;
      }
    }
    return cur_A;
  }

  //https://leetcode.com/problems/balanced-binary-tree/
  int depth(TreeNode* root) {
    if(!root) return 0;
    int left = depth(root->left);
    if (left == -1) {
      return -1;
    }
    int right = depth(root->right);
    if (right == -1) {
      return -1;
    }

    if (std::abs(left - right) > 1) return -1;
    return 1 + std::max(left, right);
  }
  bool isBalanced(TreeNode* root) {
    return depth(root) != -1;
  }

  //https://leetcode.com/problems/maximum-depth-of-n-ary-tree/
  int maxDepth(Node* root) {
    if (!root) return 0;
    int depth = 0;
    for (auto child : root->children) {
      depth = max(depth, maxDepth(child));
    }
    return depth + 1;
  }

  //https://leetcode.com/problems/populating-next-right-pointers-in-each-node/
//    Node* connect(Node* root) {
//      if (root == nullptr) return root;
//      queue<Node*> q{};
//      q.emplace(root);
//      while (!q.empty()) {
//        int len = q.size();
//        Node* pre = nullptr;
//        while (len--) {
//          Node* cur = q.front();
//          q.pop();
//          cur->next = pre;
//          pre = cur;
//
//          if (cur->right != nullptr) q.emplace(cur->right);
//          if (cur->left != nullptr) q.emplace(cur->left);
//        }
//      }
//      return root;
//    }

//  Node *connect(Node *root) {
//    Node *pre = root, *cur;
//    while (pre && pre->left) {
//      cur = pre; //cur用于level遍历, pre为每层最左边节点
//      while (cur) {
//        cur->left->next = cur->right;
//        if (cur->next) cur->right->next = cur->next->left;
//        cur = cur->next;
//      }
//      pre = pre->left;
//    }
//    return root;
//  }



  void test(vector<int>& vec) {
    vec[0] = -1;
  }
  void process_value(string& i) {
    i[0] = 'a';
    cout << "LValue process " << i << endl;
  }

  void process_value(string&& i) {
    i[0] = 'a';
    cout << "RValue process " << i << endl;
  }
};

//  std::numeric_limits<int>::max();
int main() {
  auto* s = new Solution();
  vector<vector<int>> grid{vector<int>{1,1,0,0,0},vector<int>{1,1,0,0,0},vector<int>{0,0,0,1,1},vector<int>{0,0,0,1,1}};
  cout << s->numDistinctIslands(grid) << endl;
}