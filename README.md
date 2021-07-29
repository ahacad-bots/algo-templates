

# 常用头部处理

## 快速输入输出

```cpp
template <typename T>
inline T read() {
    char ch = getchar();
    T x = 0, f = 1;
    while (ch < '0' || ch > '9') {
        if (ch == '-') f = -1;
        ch = getchar();
    }
    while ('0' <= ch && ch <= '9') {
        x = x * 10 + ch - '0';
        ch = getchar();
    }
    return x * f;
}
template <class T>
inline void write(T x) {
    if (x < 0) x = -x, putchar('-');  // 负数输出
    static T sta[35];
    T top = 0;
    do {
        sta[top++] = x % 10, x /= 10;
    } while (x);
    while (top) putchar(sta[--top] + '0');
}
```

## defines
 
```cpp
#define rep(i, x, y) for (auto i = (x); i <= (y); i++)
#define dep(i, x, y) for (auto i = (x); i >= (y); i--)

typedef long long ll;
```

# 

## 快速幂

```cpp
ll pow(ll base, ll power, ll p) {
    ll result = 1;
    while (power > 0) {
        if (power & 1) {
            result = result * base % p;
        }
        power >>= 1;
        base = (base * base) % p;
    }
    return result;
}
```

## gcdlcd

```cpp
ll gcd(ll a, ll b) {
    while (b ^= a ^= b ^= a %= b)
        ;
    return a;
}
ll lcd(ll a, ll b) { return a * b / gcd(a, b); }
```

# 树

## 线段树 (segment tree)

树写法，比较直观。

```cpp
struct node {
    int s, e, m;
    long long val = 0;
    long long lazy = 0;
    node *l, *r;

    node(int S, int E) {
        s = S, e = E, m = (s + e) / 2;
        if (s != e) {
            l = new node(s, m);
            r = new node(m + 1, e);
        }
    }
    void apply(long long L) {
        val += L * (e - s + 1);
        lazy += L;
    }
    void push() {
        if (s == e) return;
        l->apply(lazy);
        r->apply(lazy);
        lazy = 0;
    }
    void update(int S, int E, long long L) {
        push();
        if (S <= s && e <= E) {
            apply(L);
            return;
        }
        if (S <= m) l->update(S, E, L);
        if (E > m) r->update(S, E, L);
        val = l->val + r->val;
    }
    long long query(int S, int E) {
        if (S <= s && e <= E) {
            return val;
        }
        push();
        ll res = 0;
        if (S <= m) res += l->query(S, E);
        if (E > m) res += r->query(S, E);
        return res;
    }
};
```

数组写法 TODO

```cpp

```

## 树状数组 (BIT)

```cpp
struct bit {
    int t1[maxn], t2[maxn], n; // !!!: 注意要初始化 n 为你想要设置的数量上限
    
    void _add(int k, int v) {
      int v1 = k * v;
      while (k <= n) {
        t1[k] += v, t2[k] += v1;
        k += k & -k;
      }
    }
    int _getsum(int *t, int k) {
      int ret = 0;
      while (k) {
        ret += t[k];
        k -= k & -k;
      }
      return ret;
    }
    void add(int l, int r, int v) {
      _add(l, v), _add(r + 1, -v);  // 将区间加差分为两个前缀加
    }
    long long getsum(int l, int r) {
      return (r + 1ll) * _getsum(t1, r) - 1ll * l * _getsum(t1, l - 1) -
             (_getsum(t2, r) - _getsum(t2, l - 1));
    }
};
```



## 树链剖分

- $fa(x)$: 父节点
- $dep(x)$: 节点深度
- $siz(x)$: 子树节点个数
- $son(x)$: 重子节点
- $top(x)$: x 所在重链的顶部节点
- $dfn(x)$: x 的 dfs 序、线段树中编号
- $rnk(x)$: dfs 序对应的节点编号，dfn 反函数

```cpp
const int maxn = 1e5 + 5;

int cnt, fa[maxn], son[maxn], siz[maxn], dep[maxn], dfn[maxn], rnk[maxn],
    top[maxn];
int cur, h[maxn], p[maxn], nxt[maxn];

inline void add_edge(int x, int y) {
    cur++;
    nxt[cur] = h[x];
    h[x] = cur;
    p[cur] = y;
}
void dfs1(int o) {
    son[o] = -1;                         // 重儿子初始化
    siz[o] = 1;                          // 子树节点数量初始化
    for (int j = h[o]; j; j = nxt[j]) {  // 遍历儿子
        if (!dep[p[j]]) {                // 还没有访问过、故深度为0
            dep[p[j]] = dep[o] + 1;
            fa[p[j]] = o;
            dfs1(p[j]);
            siz[o] += siz[p[j]];  // 更新子树节点数量
            if (son[o] == -1 || siz[p[j]] > siz[son[o]])
                son[o] = p[j];  // 更新重子节点
        }
    }
}
void dfs2(int o, int t) {
    top[o] = t;  // 重链顶部节点更新
    cnt++;
    dfn[o] = cnt;
    rnk[cnt] = o;
    if (son[o] == -1) return;
    dfs2(son[o], t);  // 优先遍历重子节点
    for (int j = h[o]; j; j = nxt[j])
        if (p[j] != son[o] && p[j] != fa[o]) dfs2(p[j], p[j]);
}

int lca(int u, int v) { // 找公共祖先
  while (top[u] != top[v]) {
    if (dep[top[u]] > dep[top[v]])
      u = fa[top[u]];
    else
      v = fa[top[v]];
  }
  return dep[u] > dep[v] ? v : u;
}

void init() {
    dep[1] = 1; // 注意根节点深度要初始化为 1
    fa[1] = 1;
    dfs1(1);
    dfs2(1, 1);
}
```

# 图


## 匈牙利算法（二分图最大匹配）


