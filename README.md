
# 常用头部处理

## 直接上 Template

```cpp
#include <bits/stdc++.h>
using namespace std;

#define rep(i, x, y) for (auto i = (x); i <= (y); i++)
#define dep(i, x, y) for (auto i = (x); i >= (y); i--)
#define ____ puts("\n_______________\n") 
#define debug(x)  cout<< #x << " => " << (x) << endl

typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pii;

void init() {
    //
}
void solve() {
    //
}
void clear() {
    //
}

int main() {
    ios::sync_with_stdio(false), cin.tie(0), cout.tie(0);
}
```


## defines
 
```cpp
#define rep(i, x, y) for (auto i = (x); i <= (y); i++)
#define dep(i, x, y) for (auto i = (x); i >= (y); i--)

typedef long long ll;
```

## Debug 专用

```cpp
#define ____ puts("\n_______________\n") 
#define debug(x) ____; cout<< #x << " => " << (x) << endl
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

区间查询单点修改

```cpp
#define lowbit(x) ((x) & -(x))
int arr[N]; void add(int x){ while(x <= N){ ++arr[x]; x += lowbit(x); } }
int qry(int x){int sum = 0; while(x) {sum += arr[x]; x -= lowbit(x);} return sum;}
int qry(int l , int r){return qry(r) - qry(l - 1);}
```

区间查询区间修改

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

## 求树的直径

两次 dfs 方法求直径

```cpp
const int N = 10000 + 10;

int n, c, d[N];
vector<int> E[N];

void dfs(int u, int fa) {
  for (int v : E[u]) {
    if (v == fa) continue;
    d[v] = d[u] + 1;
    if (d[v] > d[c]) c = v;
    dfs(v, u);
  }
}

int main() {
  scanf("%d", &n);
  for (int i = 1; i < n; i++) {
    int u, v;
    scanf("%d %d", &u, &v);
    E[u].push_back(v), E[v].push_back(u);
  }
  dfs(1, 0);
  d[c] = 0, dfs(c, 0);
  printf("%d\n", d[c]);
  return 0;
}
```

## 笛卡尔树

每个节点是一个 $(k, v)$ 二元组，一个满足二叉搜索树另一个满足堆。

单调栈超基本笛卡尔建树模板

```cpp
#include <bits/stdc++.h>
#define re register
#define il inline
#define LL long long
using namespace std;
template <typename T>
il void read(T &ff) {
    T rr = 1;
    ff = 0;
    char ch = getchar();
    while (!isdigit(ch)) {
        if (ch == '-') rr = -1;
        ch = getchar();
    }
    while (isdigit(ch)) {
        ff = (ff << 1) + (ff << 3) + (ch ^ 48);
        ch = getchar();
    }
    ff *= rr;
}

const int N = 1e7 + 7;
int n, a[N], stk[N], ls[N], rs[N];
LL L, R;
signed main() {
    read(n);
    for (int i = 1, pos = 0, top = 0; i <= n;
         ++i) {  //这是按下标顺序插入元素的代码
        read(a[i]);
        //除了上述的维护左右儿子就是普通单调栈啦
        pos = top;
        while (pos && a[stk[pos]] > a[i]) pos--;
        if (pos) rs[stk[pos]] = i;
        if (pos < top) ls[i] = stk[pos + 1];
        stk[top = ++pos] = i;
    }
    for (int i = 1; i <= n; ++i)
        L ^= 1LL * i * (ls[i] + 1), R ^= 1LL * i * (rs[i] + 1);
    printf("%lld %lld", L, R);
    return 0;
}

```

笛卡尔树算最大子矩阵

```cpp
#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const int N = 100000 + 10, INF = 0x3f3f3f3f;

struct node {
  int idx, val, par, ch[2];
  friend bool operator<(node a, node b) { return a.idx < b.idx; }
  void init(int _idx, int _val, int _par) {
    idx = _idx, val = _val, par = _par, ch[0] = ch[1] = 0;
  }
} tree[N];

int root, top, stk[N];
ll ans;
int cartesian_build(int n) {  //建树，满足小根堆性质
  for (int i = 1; i <= n; i++) {
    int k = i - 1;
    while (tree[k].val > tree[i].val) k = tree[k].par;
    tree[i].ch[0] = tree[k].ch[1];
    tree[k].ch[1] = i;
    tree[i].par = k;
    tree[tree[i].ch[0]].par = i;
  }
  return tree[0].ch[1];
}
int dfs(int x) {  //一次dfs更新答案就可以了
  if (!x) return 0;
  int sz = dfs(tree[x].ch[0]);
  sz += dfs(tree[x].ch[1]);
  ans = max(ans, (ll)(sz + 1) * tree[x].val);
  return sz + 1;
}
int main() {
  int n, hi;
  while (scanf("%d", &n), n) {
    tree[0].init(0, 0, 0);
    for (int i = 1; i <= n; i++) {
      scanf("%d", &hi);
      tree[i].init(i, hi, 0);
    }
    root = cartesian_build(n);
    ans = 0;
    dfs(root);
    printf("%lld\n", ans);
  }
  return 0;
}
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

## 最近公共祖先 (LCA)

# 图
 
## 最短路 (TODO)

### Floyd

复杂度 $O(n^3)$，一般来说用不到，但是万一呢

```cpp
for (k = 1; k <= n; k++) {
  for (x = 1; x <= n; x++) {
    for (y = 1; y <= n; y++) {
      f[x][y] = min(f[x][y], f[x][k] + f[k][y]);
    }
  }
}
```

### Bellman-Ford 与 SPFA (TODO)

### Dijkstra

Priority_queue，复杂度 $O(mlogm)$

```cpp
struct node {
    int to, next, w;
};
int head[MAXN];
node edge[MAXN];
int cnt;
void add(int x, int y, int w) {
    edge[++cnt].to = y;
    edge[cnt].next = head[x];
    edge[cnt].w = w;
    head[x] = cnt;
}

int d[MAXN];
bool vis[MAXN];
void dij(int s) {
    memset(d, 0x3f, sizeof(d));
    memset(vis, 0, sizeof(vis));
    d[s] = 0;
    priority_queue<pair<int, int> > q;
    q.push(make_pair(0, s));
    while (!q.empty()) {
        int u = q.top().second;
        q.pop();
        if (vis[u]) continue;
        vis[u] = 1;
        for (int i = head[u]; i; i = edge[i].next) {
            int to = edge[i].to, w = edge[i].w;
            if (d[to] > d[u] + w)
                d[to] = d[u] + w, q.push(make_pair(-d[to], to));
        }
    }
}
```

## 最小生成树 (TODO)

### Prim

堆优化复杂度 $O(nlogn + m)$

```cpp
int k, n, m, cnt, sum, ai, bi, ci, head[5005], dis[5005], vis[5005];
struct Edge {
    int v, w, next;
} e[400005];
void add(int u, int v, int w) {
    e[++k].v = v;
    e[k].w = w;
    e[k].next = head[u];
    head[u] = k;
}
priority_queue<pii, vector<pii>, greater<pii> > q;
void prim() {
    dis[1] = 0;
    q.push(make_pair(0, 1));
    while (!q.empty() && cnt < n) {
        int d = q.top().first, u = q.top().second;
        q.pop();
        if (vis[u]) continue;
        cnt++;
        sum += d;
        vis[u] = 1;
        for (int i = head[u]; i != -1; i = e[i].next)
            if (e[i].w < dis[e[i].v])
                dis[e[i].v] = e[i].w, q.push(make_pair(dis[e[i].v], e[i].v));
    }
}

int main() {
    memset(dis, 127, sizeof(dis));
    memset(head, -1, sizeof(head));
    scanf("%d%d", &n, &m);
    for (int i = 1; i <= m; i++) {
        scanf("%d%d%d", &ai, &bi, &ci);
        add(ai, bi, ci);
        add(bi, ai, ci);
    }
    prim();
    if (cnt == n)
        printf("%d", sum);
    else
        printf("orz");
}
```

### Kruskal

堆优化复杂度 $O(mlogm)$

```cpp
struct node {
    int u;
    int v;
    int w;
} e[maxn];
int fa[maxn], cnt, sum, num;
void add(int x, int y, int w) {
    e[++cnt].u = x;
    e[cnt].v = y;
    e[cnt].w = w;
}
bool cmp(node x, node y) { return x.w < y.w; }
int find(int x) {
    return fa[x] == x ? fa[x] : fa[x] = find(fa[x]);  //路径压缩
}
void kruskal() {
    for (int i = 1; i <= cnt; i++) {
        int x = find(e[i].u);
        int y = find(e[i].v);
        if (x == y) continue;
        fa[x] = y;
        sum += e[i].w;
        if (++num == n - 1) break;  //如果构成了一颗树
    }
}

int main() {
    n = read<int>();
    m = read<int>();
    for (int i = 1; i <= n; i++) fa[i] = i;
    while (m--) {
        int x, y, w;
        x = read<int>();
        y = read<int>();
        w = read<int>();
        add(x, y, w);
    }
    std::sort(e + 1, e + 1 + cnt, cmp);
    kruskal();
    printf("%d", sum);
    return 0;
}
```

### Borůvka

复杂度 $O(mlogn)$

```cpp
#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;

const int MaxN = 5000 + 5, MaxM = 200000 + 5;

int N, M;
int U[MaxM], V[MaxM], W[MaxM];
bool used[MaxM];
int par[MaxN], Best[MaxN];

void init() {
    scanf("%d %d", &N, &M);
    for (int i = 1; i <= M; ++i)
        scanf("%d %d %d", &U[i], &V[i], &W[i]);
}

void init_dsu() {
    for (int i = 1; i <= N; ++i)
        par[i] = i;
}

int get_par(int x) {
    if (x == par[x]) return x;
    else return par[x] = get_par(par[x]);
}

inline bool Better(int x, int y) {
    if (y == 0) return true;
    if (W[x] != W[y]) return W[x] < W[y];
    return x < y;
}

void Boruvka() {
    init_dsu();

    int merged = 0, sum = 0;

    bool update = true;
    while (update) {
        update = false;
        memset(Best, 0, sizeof Best);

        for (int i = 1; i <= M; ++i) {
            if (used[i] == true) continue;
            int p = get_par(U[i]), q = get_par(V[i]);
            if (p == q) continue;

            if (Better(i, Best[p]) == true) Best[p] = i;
            if (Better(i, Best[q]) == true) Best[q] = i;
        }

        for (int i = 1; i <= N; ++i)
            if (Best[i] != 0 && used[Best[i]] == false) {
                update = true;
                merged++; sum += W[Best[i]];
                used[Best[i]] = true;
                par[get_par(U[Best[i]])] = get_par(V[Best[i]]);
            }
    }

    if (merged == N - 1) printf("%d\n", sum);
    else puts("orz");
}

int main() {
    init();
    Boruvka();
    return 0;
}
```

### Link-cut Tree

用 Lint-cut tree [也可以做](https://www.luogu.com.cn/blog/Soulist/solution-p3366) orz

```cpp
#include <bits/stdc++.h>
using namespace std;
int read() {
    char cc = getchar();
    int cn = 0, flus = 1;
    while (cc < '0' || cc > '9') {
        if (cc == '-') flus = -flus;
        cc = getchar();
    }
    while (cc >= '0' && cc <= '9') cn = cn * 10 + cc - '0', cc = getchar();
    return cn * flus;
}
const int N = 2e5 + 5005;
#define ls(x) t[x].son[0]
#define rs(x) t[x].son[1]
struct LCT {
    int son[2], mx, id, fa;
    bool mark;
} t[N];
int w[N], n, m, Idnet, ans;
void pushup(int x) {  //每次下传都需要更新最大点权，
    t[x].id = x, t[x].mx = w[x];
    if (t[ls(x)].mx > t[x].mx) t[x].mx = t[ls(x)].mx, t[x].id = t[ls(x)].id;
    if (t[rs(x)].mx > t[x].mx) t[x].mx = t[rs(x)].mx, t[x].id = t[rs(x)].id;
}
bool isroot(int x) { return (rs(t[x].fa) != x) && (ls(t[x].fa) != x); }
void pushmark(int x) {  //下传翻转标记
    if (t[x].mark) {
        t[x].mark = 0, t[ls(x)].mark ^= 1, t[rs(x)].mark ^= 1;
        swap(ls(x), rs(x));
    }
}
void rotate(int x) {  //旋转
    int f = t[x].fa, ff = t[f].fa, qwq = (rs(f) == x);
    t[x].fa = ff;
    if (!isroot(f)) t[ff].son[(rs(ff) == f)] = x;  //如果父亲不为根才改爷爷
    t[t[x].son[qwq ^ 1]].fa = f, t[f].son[qwq] = t[x].son[qwq ^ 1], t[f].fa = x,
                     t[x].son[qwq ^ 1] = f;
    pushup(f), pushup(x);
}
int st[N];
void Splay(int x) {
    int top = 0, now = x;
    st[++top] = now;
    while (!isroot(now)) st[++top] = (now = t[now].fa);
    while (top) pushmark(st[top--]);
    while (!isroot(x)) {
        int f = t[x].fa, ff = t[f].fa;
        if (!isroot(f)) ((rs(ff) == f) ^ (rs(f) == x)) ? rotate(x) : rotate(f);
        rotate(x);
    }
}
void access(int x) {
    for (int y = 0; x; y = x, x = t[y].fa) Splay(x), t[x].son[1] = y, pushup(x);
}
void makeroot(int x) { access(x), Splay(x), t[x].mark ^= 1, pushmark(x); }
int findroot(int x) {
    access(x), Splay(x), pushmark(x);
    while (ls(x)) pushmark(x = ls(x));
    return x;
}
void split(int x, int y) { makeroot(x), access(y), Splay(y); }
bool check(int x, int y) {  //判断两个点是否联通
    makeroot(x);
    return findroot(y) != x;
}
void link(int x, int y) {  // link的前提是这两个点联通，所以没有判断
    makeroot(x);
    t[x].fa = y;
}
signed main() {
    n = read(), m = read();
    Idnet = n;  // Idnet表示当前边

    int x, y, z, now;

    for (int i = 1; i <= m; ++i) {
        x = read(), y = read(), z = read();

        ++Idnet,
            w[Idnet] = z;  //表示编号为Idnet的边（也是LCT中的树点）的点权变成z

        if (check(x, y))
            link(x, Idnet), link(Idnet, y),
                ans += z;  //如果两个不在同一个联通快里面，直接连边，并更新答案
        else {
            split(x, y),  //把x-y的路径先拉出来
                now = t[y].id;
            if (t[now].mx <= z) continue;
            ans += (z - t[now].mx), Splay(t[y].id);  //先把这个点旋上去
            t[ls(now)].fa = t[rs(now)].fa = 0;  //子不认父，就是断边
            link(x, Idnet), link(Idnet, y);     //再连边
        }
    }
    printf("%d\n", ans);
    return 0;
}
```

## （严格）次小生成树

次小生成树与最小生成树差的只是一条边，枚举插入每条非树边，然后在环上删除最大边。

```cpp
#include <bits/stdc++.h>
#define INF 2100000001
#define M 300003
#define N 100003
#define LL long long
using namespace std;

int read() {
    int f = 1, x = 0;
    char s = getchar();
    while (s < '0' || s > '9') {
        if (s == '-') f = -1;
        s = getchar();
    }
    while (s >= '0' && s <= '9') {
        x = x * 10 + s - '0';
        s = getchar();
    }
    return x * f;
}
struct EDGE {
    int x, y, z, flagg;
} w[M];
struct edgee {
    int to, nextt, val;
} e[M];
int tot = 0, m, n, minn = INF;
LL ans = 0;
int f[N][22], max1[N][22], g[N][22], fa[N], head[N], dep[N];
bool cmp(const EDGE &a, const EDGE &b) { return a.z < b.z; }
int getfa(int x) {
    if (fa[x] == x) return x;
    return fa[x] = getfa(fa[x]);
}
void add(int a, int b, int v) {
    tot++;
    e[tot].to = b;
    e[tot].nextt = head[a];
    e[tot].val = v;
    head[a] = tot;
}
void kruscal() {
    int q = 1;
    sort(w + 1, w + m + 1, cmp);
    for (int i = 1; i <= n; ++i) fa[i] = i;
    for (int i = 1; i <= m; ++i) {
        int s1 = getfa(w[i].x);
        int s2 = getfa(w[i].y);
        if (s1 != s2) {
            ans += w[i].z;
            w[i].flagg = 1;
            q++;
            fa[s1] = s2;
            add(w[i].x, w[i].y, w[i].z);
            add(w[i].y, w[i].x, w[i].z);
        }
        if (q == n) break;
    }
}
void dfs(int x) {
    for (int i = head[x]; i; i = e[i].nextt) {
        int v = e[i].to;
        if (v == f[x][0]) continue;
        f[v][0] = x;
        max1[v][0] = e[i].val;
        dep[v] = dep[x] + 1;
        for (int j = 1; j <= 20; ++j) {
            if (dep[v] < (1 << j))
                break;  //注意：如果深度小于向上走的步数就可以break掉了
            f[v][j] = f[f[v][j - 1]][j - 1];  // f是向上走到达的点
            max1[v][j] =
                max(max1[v][j - 1], max1[f[v][j - 1]][j - 1]);  // max1是最大边
            if (max1[v][j - 1] == max1[f[v][j - 1]][j - 1])
                g[v][j] = max(g[v][j - 1], g[f[v][j - 1]][j - 1]);  // g是次大边
            else {
                g[v][j] = min(max1[v][j - 1], max1[f[v][j - 1]][j - 1]);
                g[v][j] = max(g[v][j], g[f[v][j - 1]][j - 1]);
                g[v][j] = max(g[v][j - 1], g[v][j]);
            }
        }
        dfs(v);
    }
}
int LCA(int u, int x) {
    if (dep[u] < dep[x]) swap(u, x);
    for (int i = 20; i >= 0; --i)
        if (dep[f[u][i]] >= dep[x]) u = f[u][i];
    if (x == u) return x;
    for (int i = 20; i >= 0; --i)
        if (f[x][i] != f[u][i]) x = f[x][i], u = f[u][i];
    return f[x][0];
}
void change(int x, int lca, int val) {
    int maxx1 = 0, maxx2 = 0;
    int d = dep[x] - dep[lca];
    for (int i = 0; i <= 20; ++i) {
        if (d < (1 << i)) break;
        if (d & (1 << i)) {
            if (max1[x][i] > maxx1) {
                maxx2 = max(maxx1, g[x][i]);
                maxx1 = max1[x][i];
            }
            x = f[x][i];
        }
    }
    if (val != maxx1)
        minn = min(minn, val - maxx1);
    else
        minn = min(minn, val - maxx2);
}
void work() {
    for (int i = 1; i <= m; ++i) {
        if (!w[i].flagg) {
            int s1 = w[i].x, s2 = w[i].y;
            int lca = LCA(s1, s2);
            change(s1, lca, w[i].z);
            change(s2, lca, w[i].z);
        }
    }
}
int main() {
    n = read();
    m = read();
    for (int i = 1; i <= m; ++i) {
        w[i].x = read();
        w[i].y = read();
        w[i].z = read();
    }
    kruscal();
    dfs(1);
    work();
    printf("%lld\n", ans + minn);
}
```

## 匈牙利算法（二分图最大匹配） (TODO)

```cpp

```

## 网络流 (TODO)

# 字符串

## 字符串哈希

按进制哈希

```cpp
ull base = 131;
int prime = 233317;
ull mod=212370440130137957ll;
ull hashe(char s[]) {
    int len = strlen(s);
    ull ans = 0;
    for (int i = 0; i < len; i++) ans = (ans * base + (ull)s[i]) % mod + prime;
    return ans;
}
```


BKDRHash

```cpp
unsigned int BKDRHash(char* str) {
    unsigned int seed = 131;  // 31 131 1313 13131 131313 etc..
    unsigned int hash = 0;
    while (*str) {
        hash = hash * seed + (*str++);
    }
    return (hash % mod);
}
```

## KMP

```cpp
#include <cstring>
#include <iostream>
#define MAXN 1000010
using namespace std;
int kmp[MAXN];
int la, lb, j;
char a[MAXN], b[MAXN];
int main() {
    cin >> (a + 1);
    cin >> (b + 1);
    la = strlen(a + 1);
    lb = strlen(b + 1);
    for (int i = 2; i <= lb; i++) {
        while (j && b[i] != b[j + 1]) j = kmp[j];
        if (b[j + 1] == b[i]) j++;
        kmp[i] = j;
    }
    j = 0;
    for (int i = 1; i <= la; i++) {
        while (j > 0 && b[j + 1] != a[i]) j = kmp[j];
        if (b[j + 1] == a[i]) j++;
        if (j == lb) {
            cout << i - lb + 1 << endl;
            j = kmp[j];
        }
    }

    for (int i = 1; i <= lb; i++) cout << kmp[i] << " ";
    return 0;
}
```

## Manacher

线性求最长回文串。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
using namespace std;
const int maxn = 11000002;
char dat[maxn << 1];
int p[maxn << 1], cnt, ans;
inline void qr() {
    char c = getchar();
    dat[0] = '~', dat[cnt = 1] = '|';
    while (c < 'a' || c > 'z') c = getchar();
    while (c >= 'a' && c <= 'z')
        dat[++cnt] = c, dat[++cnt] = '|', c = getchar();
}
int main() {
    qr();
    for (int t = 1, r = 0, mid = 0; t <= cnt; ++t) {
        if (t <= r) p[t] = min(p[(mid << 1) - t], r - t + 1);
        while (dat[t - p[t]] == dat[t + p[t]]) ++p[t];
        if (p[t] + t > r) r = p[t] + t - 1, mid = t;
        if (p[t] > ans) ans = p[t];
    }
    printf("%d\n", ans - 1);
    return 0;
}
```

## AC 自动机

统计每个子串出现次数

```cpp
// AC自动机加强版
#include <bits/stdc++.h>
#define maxn 1000001
using namespace std;
char s[151][maxn], T[maxn];
int n, cnt, vis[maxn], ans;
struct kkk {
    int son[26], fail, flag;
    void clear() {
        memset(son, 0, sizeof(son));
        fail = flag = 0;
    }
} trie[maxn];
void insert(char* s, int num) {
    int u = 1, len = strlen(s);
    for (int i = 0; i < len; i++) {
        int v = s[i] - 'a';
        if (!trie[u].son[v]) trie[u].son[v] = ++cnt;
        u = trie[u].son[v];
    }
    trie[u].flag = num;  //变化1：标记为第num个出现的字符串
}
queue<int> q;
void getFail() {
    for (int i = 0; i < 26; i++) trie[0].son[i] = 1;
    q.push(1);
    trie[1].fail = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        int Fail = trie[u].fail;
        for (int i = 0; i < 26; i++) {
            int v = trie[u].son[i];
            if (!v) {
                trie[u].son[i] = trie[Fail].son[i];
                continue;
            }
            trie[v].fail = trie[Fail].son[i];
            q.push(v);
        }
    }
}
void query(char* s) {
    int u = 1, len = strlen(s);
    for (int i = 0; i < len; i++) {
        int v = s[i] - 'a';
        int k = trie[u].son[v];
        while (k > 1) {
            if (trie[k].flag)
                vis[trie[k].flag]++;  //如果有模式串标记，更新出现次数
            k = trie[k].fail;
        }
        u = trie[u].son[v];
    }
}
void clear() {
    for (int i = 0; i <= cnt; i++) trie[i].clear();
    for (int i = 1; i <= n; i++) vis[i] = 0;
    cnt = 1;
    ans = 0;
}
int main() {
    while (1) {
        scanf("%d", &n);
        if (!n) break;
        clear();
        for (int i = 1; i <= n; i++) {
            scanf("%s", s[i]);
            insert(s[i], i);
        }
        scanf("%s", T);
        getFail();
        query(T);
        for (int i = 1; i <= n; i++) ans = max(vis[i], ans);  //最后统计答案
        printf("%d\n", ans);
        for (int i = 1; i <= n; i++)
            if (vis[i] == ans) printf("%s\n", s[i]);
    }
}
```

AC 自动机 topu 优化

```cpp
#include <bits/stdc++.h>
#define maxn 2000001
using namespace std;
char s[maxn], T[maxn];
int n, cnt, vis[200051], ans, in[maxn], Map[maxn];
struct kkk {
    int son[26], fail, flag, ans;
} trie[maxn];
queue<int> q;
void insert(char* s, int num) {
    int u = 1, len = strlen(s);
    for (int i = 0; i < len; ++i) {
        int v = s[i] - 'a';
        if (!trie[u].son[v]) trie[u].son[v] = ++cnt;
        u = trie[u].son[v];
    }
    if (!trie[u].flag) trie[u].flag = num;
    Map[num] = trie[u].flag;
}
void getFail() {
    for (int i = 0; i < 26; i++) trie[0].son[i] = 1;
    q.push(1);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        int Fail = trie[u].fail;
        for (int i = 0; i < 26; ++i) {
            int v = trie[u].son[i];
            if (!v) {
                trie[u].son[i] = trie[Fail].son[i];
                continue;
            }
            trie[v].fail = trie[Fail].son[i];
            in[trie[v].fail]++;
            q.push(v);
        }
    }
}
void topu() {
    for (int i = 1; i <= cnt; ++i)
        if (in[i] == 0) q.push(i);  //将入度为0的点全部压入队列里
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        vis[trie[u].flag] = trie[u].ans;  //如果有flag标记就更新vis数组
        int v = trie[u].fail;
        in[v]--;  //将唯一连出去的出边fail的入度减去（拓扑排序的操作）
        trie[v].ans += trie[u].ans;  //更新fail的ans值
        if (in[v] == 0) q.push(v);   //拓扑排序常规操作
    }
}
void query(char* s) {
    int u = 1, len = strlen(s);
    for (int i = 0; i < len; ++i) u = trie[u].son[s[i] - 'a'], trie[u].ans++;
}
int main() {
    scanf("%d", &n);
    cnt = 1;
    for (int i = 1; i <= n; ++i) {
        scanf("%s", s);
        insert(s, i);
    }
    getFail();
    scanf("%s", T);
    query(T);
    topu();
    for (int i = 1; i <= n; ++i) printf("%d\n", vis[Map[i]]);
}
```

## Trie 树

### 基本字符串查找

```cpp
struct trie {
  int nex[100000][26], cnt = 0;
  bool exist[100000];  // 该结点结尾的字符串是否存在

  void insert(char *s, int l) {  // 插入字符串
    int p = 0;
    for (int i = 0; i < l; i++) {
      int c = s[i] - 'a';
      if (!nex[p][c]) nex[p][c] = ++cnt;  // 如果没有，就添加结点
      p = nex[p][c];
    }
    exist[p] = 1;
  }
  bool find(char *s, int l) {  // 查找字符串
    int p = 0;
    for (int i = 0; i < l; i++) {
      int c = s[i] - 'a';
      if (!nex[p][c]) return false;
      p = nex[p][c];
    }
    return true;
  }
};
```

### 维护异或极值

```cpp
const int N = 100010;

int head[N], nxt[N << 1], to[N << 1], weight[N << 1], cnt;
int n, dis[N], ch[N << 5][2], tot = 1, ans;

void insert(int x) {
    for (int i = 30, u = 1; i >= 0; --i) {
        int c = ((x >> i) & 1);
        if (!ch[u][c]) ch[u][c] = ++tot;
        u = ch[u][c];
    }
}
void get(int x) {
    int res = 0;
    for (int i = 30, u = 1; i >= 0; --i) {
        int c = ((x >> i) & 1);
        if (ch[u][c ^ 1]) {
            u = ch[u][c ^ 1];
            res |= (1 << i);
        } else
            u = ch[u][c];
    }
    ans = std::max(ans, res);
}
void add(int u, int v, int w) {
    nxt[++cnt] = head[u];
    head[u] = cnt;
    to[cnt] = v;
    weight[cnt] = w;
}
void dfs(int u, int fa) {
    insert(dis[u]);
    get(dis[u]);
    for (int i = head[u]; i; i = nxt[i]) {
        int v = to[i];
        if (v == fa) continue;
        dis[v] = dis[u] ^ weight[i];
        dfs(v, u);
    }
}
int main() {
    scanf("%d", &n);

    for (int i = 1; i < n; ++i) {
        int u, v, w;
        scanf("%d%d%d", &u, &v, &w);
        add(u, v, w);
        add(v, u, w);
    }

    dfs(1, 0);

    printf("%d", ans);

    return 0;
}
```


# 数学

## 快速傅里叶变换 (FFT)

```cpp
const int MAXN = 1e7 + 10;
const double Pi = acos(-1.0);
struct Complex {
    double x, y;
    Complex(int _x, int _y) { x = _x, y = _y; }
    Complex(double _x = 0, double _y = 0) { x = _x, y = _y; }
    friend Complex operator+(const Complex &a, const Complex &b) {
        return ((Complex){a.x + b.x, a.y + b.y});
    }
    friend Complex operator-(const Complex &a, const Complex &b) {
        return ((Complex){a.x - b.x, a.y - b.y});
    }
    friend Complex operator*(const Complex &a, const Complex &b) {
        return ((Complex){a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x});
    }
    friend Complex operator*(const Complex &a, const double &val) {
        return ((Complex){a.x * val, a.y * val});
    }
} f[MAXN], g[MAXN], p[MAXN];
int n, m, lim = 1, maxn, rev[MAXN], a[MAXN], b[MAXN];
inline void FFT(Complex *A, int opt) {
    for (int i = 0; i < lim; i++)
        if (i < rev[i]) swap(A[i], A[rev[i]]);
    for (int mid = 1; mid < lim; mid <<= 1) {
        Complex Wn = ((Complex){cos(Pi / (double)mid),
                                (double)opt * sin(Pi / (double)mid)});
        for (int j = 0; j < lim; j += (mid << 1)) {
            Complex W = ((Complex){1, 0});
            for (int k = 0; k < mid; k++, W = W * Wn) {
                Complex x = A[j + k], y = W * A[j + k + mid];
                A[j + k] = x + y;
                A[j + k + mid] = x - y;
            }
        }
    }
}
void init() {
    int l = 0;
    lim = 1;
    while (lim <= n + m) lim <<= 1, l++;
    rep(i, 0, lim - 1) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (l - 1));
}

int main() {
    cin >> n >> m;
    rep(i, 0, n) cin >> f[i].x;
    rep(i, 0, m) cin >> g[i].x;
    init();
    FFT(f, 1);
    FFT(g, 1);
    rep(i, 0, lim) f[i] = f[i] * g[i];
    FFT(f, -1);
    rep(i, 0, n + m) { cout << (int)(f[i].x / lim + 0.5) << " "; }

    return 0;
}
```

TODO: 快速傅立叶变换求 string 匹配

## 筛法

### 埃氏筛

复杂度 $O(n loglogn)$

```cpp
int Eratosthenes(int n) {
  int p = 0;
  for (int i = 0; i <= n; ++i) is_prime[i] = 1;
  is_prime[0] = is_prime[1] = 0;
  for (int i = 2; i * i <= n; ++i) {
    if (is_prime[i]) {
      prime[p++] = i;  // prime[p]是i,后置自增运算代表当前素数数量
      if ((long long)i * i <= n)
        for (int j = i * i; j <= n; j += i)
          // 因为从 2 到 i - 1 的倍数我们之前筛过了，这里直接从 i
          // 的倍数开始，提高了运行速度
          is_prime[j] = 0;  // 是i的倍数的均不是素数
    }
  }
  return p;
}
```

### Euler 筛

线性时间复杂度，因子筛不重复。欧拉筛可以顺便求出每个数字的因子个数。

```cpp
const int MAXN = 1e8 + 10;
int divisions[MAXN] // euler 筛还可以顺便算出每个数字的因子个数
int prime[MAXN];  //保存素数，注意下面的实现 prime 从 0 开始
bool vis[MAXN];   //初始化
int Prime(int n) {
    int cnt = 0;
    memset(vis, 0, sizeof(vis));
    for (int i = 2; i < n; i++) {
        if (!vis[i]) prime[cnt++] = i, divisions[i] = 1;
        for (int j = 0; j < cnt && i * prime[j] < n; j++) {
            vis[i * prime[j]] = 1, divisions[i * prime[j]] = divisions[i] + 1;
            if (i % prime[j] == 0)  //关键
                break;
        }
    }
    return cnt;  //返回小于n的素数的个数
}
```



## BSGS

```cpp
long long bsgs(long long a, long long b, long long p) {  // bsgs
    map<long long, long long> hash;
    hash.clear();  //建立一个Hash表
    b %= p;
    long long t = sqrt(p) + 1;
    for (long long i = 0; i < t; ++i)
        hash[(long long)b * power(a, i, p) % p] =
            i;  //将每个j对应的值插入Hash表
    a = power(a, t, p);
    if (!a) return b == 0 ? 1 : -1;  //特判
    for (long long i = 1; i <= t; ++i) {  //在Hash表中查找是否有i对应的j值
        long long val = power(a, i, p);
        int j = hash.find(val) == hash.end() ? -1 : hash[val];
        if (j >= 0 && i * t - j >= 0) return i * t - j;
    }
    return -1;  //无解返回-1
}
```

## 快速幂

```cpp
ll power(ll base, ll power, ll p) {
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

## 龟速乘

快速幂可能被毒瘤卡 long long，用龟速乘，牺牲速度换取正确性

```cpp
long long quick_mul(long long x,long long y,long long mod) 
{
	long long ans=0;
	while(y!=0){
		if(y&1==1)ans+=x,ans%=mod;
		x=x+x,x%=mod;
		y>>=1; 
	}
	return ans;
}

long long quick_pow(long long x,long long y,long long mod)
{
	long long sum=1;
    while(y!=0){
         if(y&1==1)sum=quick_mul(sum,x,mod),sum%=mod;
    	     x=quick_mul(x,x,mod),x%=mod;
         	 y=y>>1;
    }
    return sum;
}
```

## 快速乘 (FIXME)

09 集训队论文，速度 $O(1)$

```cpp
LL mul(LL a, LL b, LL P){
    LL L = a * (b >> 25LL) % P * (1LL << 25) % P;
    LL R = a * (b & ((1LL << 25) - 1)) % P;
    return (L + R) % P;
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

## Lucas 定理

算组合数，要求模数不太大，$1e5$ 

```cpp
long long Lucas(long long n, long long m, long long p) {
  if (m == 0) return 1;
  return (C(n % p, m % p, p) * Lucas(n / p, m / p, p)) % p;
}
```

## 中国剩余定理

```cpp
LL CRT(int k, LL* a, LL* r) {
  LL n = 1, ans = 0;
  for (int i = 1; i <= k; i++) n = n * r[i];
  for (int i = 1; i <= k; i++) {
    LL m = n / r[i], b, y;
    exgcd(m, r[i], b, y);  // b * m mod r[i] = 1
    ans = (ans + a[i] * m * b % mod) % mod;
  }
  return (ans % mod + mod) % mod;
}
```

## 康拖展开

求一个排列在全排列字典序中第几个。公式 $\sum_{i=1}^n sum_{a_i} \times (n - i)!$，利用树状数组加速 $sum$ 计算，$sum$ 表示有多少个数比自己小。

```cpp
#include <cstdio>
#include <iostream>
using namespace std;
typedef long long ll;
int n;

ll tree[1000005];  //树状数组
int lowbit(int x) { return x & -x; }
void update(int x, int y) {
    while (x <= n) {
        tree[x] += y;
        x += lowbit(x);
    }
}
ll query(int x) {
    ll sum = 0;
    while (x) {
        sum += tree[x];
        x -= lowbit(x);
    }
    return sum;
}

const ll wyx = 998244353;  //懒人专用
ll jc[1000005] = {1, 1};   //存阶乘的数组
int a[1000005];            //存数
int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) {  //预处理阶乘数组和树状数组
        jc[i] = (jc[i - 1] * i) % wyx;
        update(i, 1);
    }
    ll ans = 0;
    for (int i = 1; i <= n; i++) {
        scanf("%d", &a[i]);
        ans = (ans + ((query(a[i]) - 1) * jc[n - i]) % wyx) % wyx;  //计算ans
        update(a[i], -1);  //把a[i]变成0（原来是1，减1不就是0嘛）
    }
    printf("%lld", ans + 1);
    return 0;
}
```

## 乘法逆元

利用 Fermat 小定理用快速幂算逆元，注意要求 $p$ 必须是素数。

```cpp
ll x = power(a, b - 2, mod);
```

线性递推求逆元

```cpp
inv[1] = 1;
for (int i = 2; i <= n; ++i) {
  inv[i] = (long long)(p - p / i) * inv[p % i] % p;
}
```

## 高斯消元 (TODO)

## 欧拉回路

奇数个点完全图递推求欧拉回路。

```cpp
void dfs(int x, int y) {
    int cnt = 2;
    edges.push_back({1, x});
    while (cnt < x) {
        edges.push_back({x, cnt});
        edges.push_back({cnt, y});

        cnt++;
        edges.push_back({y, cnt});
        edges.push_back({cnt, x});

        cnt++;
    }
    edges.push_back({x, y});
    edges.push_back({y, 1});
}

for (int i = 1; i < n; i += 2) {
    dfs(i + 1, i + 2);
}

```

另一种 nb 的构造方法，旋转构造欧拉回路

```cpp
void solve() {
	vector<int> euler(1, n-1);
	for(int i = 0; i < n/2; ++i) {
		int sgn = 1, ct = i;
		for(int d = 1; d < n; ++d) {
			euler.push_back(ct);
			ct = (ct + sgn*d + n-1) % (n-1);
			sgn *= -1;
		}
		euler.push_back(n-1);
	}
}
```

## 矩阵 (FIXME)

```cpp
struct Matrix{ 
    static const int N=511,P=998244353;
    int n,a[N][2*N]; bool t;
    
    void In(ll a[][::N],int n) {
        this->n=n;
        rep(i,1,n+1) rep(j,1,n+1) this->a[i][j]=a[i][j], this->a[i][n+j]=0;
        rep(i,1,n+1) this->a[i][i+n]=1;
    }
    
    void Out(ll b[][::N]) {
        rep(i,1,n+1) rep(j,1,n+1) b[i][j]=a[i][n+j];
    }
    
    void Print() {
        if (t) printf("Inv Is:\n"); else { printf("Not Invertable!\n"); return; }
        rep(i,1,n+1) rep(j,1,n+1) printf("%d%c",a[i][n+j]," \n"[j==n]);
    }
    
    bool getinv(){
        rep(i,1,n+1){
            if(a[i][i]==0){
                rep(j,i,n+1) if(a[j][i]) swap(a[i],a[j]);
                if(!a[i][i]) return 0;
            }
            int s=a[i][i];
            rep(j,1,n+n+1) a[i][j]=1ll*a[i][j]*Pow(s,P-2)%P;
            rep(j,1,n+1) {
                if(i==j) continue;
                s=1ll*a[j][i]*Pow(a[i][i],P-2)%P;
                rep(k,1,n+n+1) a[j][k]=(a[j][k]-1ll*a[i][k]*s)%P;
            }
        }
        rep(i,1,n+1) rep(j,1,n+n+1) a[i][n+j]=(a[i][n+j]+P)%P;
        return 1;
    }
    
    bool Solve(ll a[][::N],int n,ll b[][::N]) {
        In(a,n),t=getinv(),Out(b); return t;
    }
    
    bool Check(ll a[][::N],ll b[][::N],int n) {
        static int c[::N][::N];
        memset(c,0,sizeof(c));
        rep(i,1,n+1) rep(k,1,n+1) if (a[i][k]) 
            rep(j,1,n+1) if (b[k][j]) c[i][j]+=a[i][k]*b[k][j]%P,c[i][j]%=P;
        rep(i,1,n+1) rep(j,1,n+1) c[i][j]=(c[i][j]+P)%P;
        rep(i,1,n+1) rep(j,1,n+1) if (c[i][j]!=(i==j)) return 0;
        return 1;
    }
}; 
```

## 矩阵快速幂

```cpp
#include <bits/stdc++.h>
using namespace std;
const int Mod = 1000000007;
struct Matrix {
    long long c[101][101];
} A, I;
long long n, k;
Matrix operator*(const Matrix &x, const Matrix &y) {
    Matrix a;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++) a.c[i][j] = 0;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++)
            for (int k = 1; k <= n; k++) {
                a.c[i][j] += x.c[i][k] * y.c[k][j] % Mod;
                a.c[i][j] %= Mod;
            }
    return a;
}

int main() {
    cin >> n >> k;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= n; j++) cin >> A.c[i][j];
    for (int i = 1; i <= n; i++) I.c[i][i] = 1;
    while (k > 0) {
        if (k % 2 == 1) I = I * A;
        A = A * A;
        k = k >> 1;
    }
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) cout << I.c[i][j] << ' ';
        cout << endl;
    }
    return 0;
}
```

# 数据结构

## ST 表

$\Theta(nlogn)$ 预处理，$\Theta(1)$ 查询，不支持修改。

```cpp
const int logn = 21;
const int maxn = 2000001;
int f[maxn][logn + 1], Logn[maxn + 1]; // f[i][j]: [i, i + 2^j - 1] 中最大值
void pre() {
  Logn[1] = 0;
  Logn[2] = 1;
  for (int i = 3; i < maxn; i++) {
    Logn[i] = Logn[i / 2] + 1;
  }
}
int main() {
  int n = read(), m = read();
  for (int i = 1; i <= n; i++) f[i][0] = read();
  pre();
  for (int j = 1; j <= logn; j++)
    for (int i = 1; i + (1 << j) - 1 <= n; i++)
      f[i][j] = max(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);
  for (int i = 1; i <= m; i++) {
    int x = read(), y = read();
    int s = Logn[y - x + 1];
    printf("%d\n", max(f[x][s], f[y - (1 << s) + 1][s]));
  }
  return 0;
}
```


# 杂

## 求逆序对

用归并排序即可，树状数组/线段树也可以做

```cpp
// C++ Version
int ans = 0 // 逆序对数量
void merge(int ll, int rr) {
  if (rr - ll <= 1) return;
  int mid = ll + (rr - ll >> 1);
  merge(ll, mid);
  merge(mid, rr);
  int p = ll, q = mid, s = ll;
  while (s < rr) {
    if (p >= mid || (q < rr && a[p] > a[q])) {
      t[s++] = a[q++];
      ans += mid - p;
    } else {
      t[s++] = a[p++];
    }
  }
  for (int i = ll; i < rr; ++i) a[i] = t[i];
}
```

## 悬线法

算最大子矩阵面积

```cpp
int main() {
  scanf("%d%d", &n, &m);
  for (int i = 1; i <= n; i++) {
    for (int j = 1; j <= m; j++) {
      l[j] = r[j] = j;
    }
    char s[3];
    for (int j = 1; j <= m; j++) {
      scanf("%s", s);
      if (s[0] == 'F')
        a[j]++;
      else if (s[0] == 'R')
        a[j] = 0;
    }
    for (int j = 1; j <= m; j++)
      while (l[j] != 1 && a[l[j] - 1] >= a[j]) l[j] = l[l[j] - 1];
    for (int j = m; j >= 1; j--)
      while (r[j] != m && a[r[j] + 1] >= a[j]) r[j] = r[r[j] + 1];
    for (int j = 1; j <= m; j++) ans = std::max(ans, (r[j] - l[j] + 1) * a[j]);
  }
  printf("%d", ans * 3);
  return 0;
}
```

## 倍增

一个经典的倍增例子。

```cpp
#include <cstdio>
using namespace std;

const int mod = 1000000007;

int modadd(int a, int b) {
    if (a + b >= mod) return a + b - mod;  // 减法代替取模，加快运算
    return a + b;
}
int vi[1000005];

int go[75][1000005];  // 将数组稍微开大以避免越界，小的一维尽量定义在前面
int sum[75][1000005];

int main() {
    int n, k;
    scanf("%d%d", &n, &k);
    for (int i = 1; i <= n; ++i) {
        scanf("%d", vi + i);
    }

    for (int i = 1; i <= n; ++i) {
        go[0][i] = (i + k) % n + 1;
        sum[0][i] = vi[i];
    }

    int logn = 31 - __builtin_clz(n);  // 一个快捷的取对数的方法
    for (int i = 1; i <= logn; ++i) {
        for (int j = 1; j <= n; ++j) {
            go[i][j] = go[i - 1][go[i - 1][j]];
            sum[i][j] = modadd(sum[i - 1][j], sum[i - 1][go[i - 1][j]]);
        }
    }

    long long m;
    scanf("%lld", &m);

    int ans = 0;
    int curx = 1;
    for (int i = 0; m; ++i) {
        if (m & (1 << i)) {  // 参见位运算的相关内容，意为 m 的第 i 位是否为 1
            ans = modadd(ans, sum[i][curx]);
            curx = go[i][curx];
            m ^= 1ll << i;  // 将第 i 位置零
        }
    }

    printf("%d\n", ans);
}
```

# 计算几何

## 扫描线 (TODO)

# C++ 相关

## 位运算

- `int __builtin_ffs (unsigned int x)`: 返回x的最后一位1的是从后向前第几位，比如7368（1110011001000）返回4。
- `int __builtin_clz (unsigned int x)`: 返回前导的0的个数。
- `int __builtin_ctz (unsigned int x)`: 返回后面的0个个数，和__builtin_clz相对。
- `int __builtin_popcount (unsigned int x)`: 返回二进制表示中1的个数。
- `int __builtin_parity (unsigned int x)`: 返回x的奇偶校验位，也就是x的1的个数模2的

# 注意！

> 十年OI一场空，不开long long见祖宗

一下常见错误：

- 没用 long long 或者 double 数据溢出
- mod 过程中数据溢出、没有用逆元计算、每一步都需要 mod，否则很有可能算法正确但是 mod 中间出错
- 题意读错，漏掉细节，看清楚到底要求什么、题目的一些细小限制

