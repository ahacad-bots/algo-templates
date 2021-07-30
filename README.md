

# 常用头部处理


## defines
 
```cpp
#define rep(i, x, y) for (auto i = (x); i <= (y); i++)
#define dep(i, x, y) for (auto i = (x); i >= (y); i--)

typedef long long ll;
```

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

再来多写一个树剖后的线段树查询，注意这里 `dfs2` 里面需要额外多一步 `nw[cnt] = w[o]` 来为树上节点初始化权重。

```cpp
#define ls u << 1
#define rs u << 1 | 1
struct node {
    int l, r, maxv, sum;
} tr[maxn << 2];
void build(int u, int l, int r) {
    tr[u] = {l, r, nw[r], nw[r]};
    if (l == r) return;
    int mid = (l + r) >> 1;
    build(ls, l, mid), build(rs, mid + 1, r);
    tr[u].sum = tr[ls].sum + tr[rs].sum;
    tr[u].maxv = max(tr[ls].maxv, tr[rs].maxv);
}
void upd(int u, int x, int v) {
    if (tr[u].l == tr[u].r) {
        tr[u].maxv = tr[u].sum = v;
        return;
    }
    int mid = (tr[u].l + tr[u].r) >> 1;
    if (x <= mid) {
        upd(ls, x, v);
    } else {
        upd(rs, x, v);
    }
    tr[u].sum = tr[ls].sum + tr[rs].sum;
    tr[u].maxv = max(tr[ls].maxv, tr[rs].maxv);
}
int query_sum(int u, int l, int r) {
    if (l <= tr[u].l && tr[u].r <= r) return tr[u].sum;
    int mid = (tr[u].l + tr[u].r) >> 1;
    int res = 0;
    if (l <= mid) res += query_sum(ls, l, r);
    if (r > mid) res += query_sum(rs, l, r);
    return res;
}
int query_max(int u, int l, int r) {
    if (l <= tr[u].l && tr[u].r <= r) return tr[u].maxv;
    int mid = (tr[u].l + tr[u].r) >> 1;
    int res = -1e9;
    if (l <= mid) res = query_max(ls, l, r);
    if (r > mid) res = max(res, query_max(rs, l, r));
    return res;
}

int qsum(int x, int y) {
    int res = 0;
    while (top[x] != top[y]) {
        if (dep[top[x]] < dep[top[y]]) swap(x, y);
        res += query_sum(1, dfn[top[x]], dfn[x]);
        x = fa[top[x]];
    }
    if (dep[x] > dep[y]) swap(x, y);
    res += query_sum(1, dfn[x], dfn[y]);
    return res;
}
int qmax(int x, int y) {
    int res = -1e9;
    while (top[x] != top[y]) {
        if (dep[top[x]] < dep[top[y]]) swap(x, y);
        res = max(res, query_max(1, dfn[top[x]], dfn[x]));
        x = fa[top[x]];
    }
    if (dep[x] > dep[y]) swap(x, y);
    res = max(res, query_max(1, dfn[x], dfn[y]));
    return res;
}
```

# 图


## 匈牙利算法（二分图最大匹配）


# 数学

## 快速傅里叶变换 (FFT)

```cpp
const double Pi = acos(-1.0);
const int maxn = 1e7 + 10;
int rev[maxn];
int limit = 1;
int l, r[maxn];
struct comp {
    double x, y;
    comp(double _x = 0, double _y = 0) { x = _x, y = _y; }
} a[maxn], b[maxn];
comp operator+(comp a, comp b) { return comp(a.x + b.x, a.y + b.y); }
comp operator-(comp a, comp b) { return comp(a.x - b.x, a.y - b.y); }
comp operator*(comp a, comp b) {
    return comp(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void fft(comp *A, int type) {
    for (int i = 0; i < limit; i++)
        if (i < r[i]) swap(A[i], A[r[i]]);     //求出要迭代的序列
    for (int mid = 1; mid < limit; mid <<= 1)  //待合并区间的中点
    {
        comp Wn(cos(Pi / mid), type * sin(Pi / mid));  //单位根
        for (int R = mid << 1, j = 0; j < limit;
             j += R)  // R是区间的右端点，j表示前已经到哪个位置了
        {
            comp w(1, 0);                              //幂
            for (int k = 0; k < mid; k++, w = w * Wn)  //枚举左半部分
            {
                comp x = A[j + k], y = w * A[j + mid + k];  //蝴蝶效应
                A[j + k] = x + y;
                A[j + mid + k] = x - y;
            }
        }
    }
}
void init() {
    rep(i, 0, n) cin >> a[i].x;
    rep(i, 0, m) cin >> b[i].x;
    while (limit <= n + m) limit <<= 1, l++;
    rep(i, 0, limit - 1) r[i] = (r[i >> 1] >> 1) | ((i & 1) << (l - 1));
}

int main() {
    ios::sync_with_stdio(0), cin.tie(0), cout.tie(0);
    cin >> n >> m;
    init();
    fft(a, 1);
    fft(b, 1);
    rep(i, 0, limit) a[i] = a[i] * b[i];
    fft(a, -1);
    rep(i, 0, n + m) { cout << (int)(a[i].x / limit + 0.5) << " "; }
    return 0;
}
```

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
