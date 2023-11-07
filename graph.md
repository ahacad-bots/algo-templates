
<!-- vim-markdown-toc GFM -->

* [图](#图)
    * [前向星建图](#前向星建图)
    * [最短路 (TODO)](#最短路-todo)
        * [Floyd](#floyd)
        * [Bellman-Ford 与 SPFA (TODO)](#bellman-ford-与-spfa-todo)
        * [Dijkstra](#dijkstra)
    * [最小生成树 (TODO)](#最小生成树-todo)
        * [Prim](#prim)
        * [Kruskal](#kruskal)
        * [Borůvka](#borvka)
        * [Link-cut Tree](#link-cut-tree)
    * [（严格）次小生成树](#严格次小生成树)
    * [Matrix tree theorem](#matrix-tree-theorem)
    * [匈牙利算法（二分图最大匹配）](#匈牙利算法二分图最大匹配)
    * [最小斯坦纳树](#最小斯坦纳树)
    * [最小树形图（朱刘算法）](#最小树形图朱刘算法)
    * [网络流](#网络流)
            * [Dinic](#dinic)
            * [Edmond-Karp](#edmond-karp)
            * [Ford Fulksonff](#ford-fulksonff)
            * [ISAP](#isap)
            * [HLPP (TODO)](#hlpp-todo)
    * [拓扑序](#拓扑序)
        * [Kahn 算法](#kahn-算法)

<!-- vim-markdown-toc -->
# 图

## 前向星建图

```cpp
int head[N],to[N],nxt[N],cnt;
void add(int x,int y) { //链式前向星
    to[++cnt]=y;
    nxt[cnt]=head[x];
    head[x]=cnt;
}
void dfs(int x) {
    for(int i=head[x];i;i=nxt[i]) { // 遍历
        int y=to[i];
        dfs(y);
    }
}
```
 
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

Prim 写法

```cpp
#include<bits/stdc++.h>
using namespace std;
#define ms(x, n) memset(x,n,sizeof(x));
typedef  long long LL;
const int inf = 1 << 30;
const LL maxn = 1010;

int N;
double w[maxn][maxn];
struct node {
    int x, y;
    int p;
    node(int xx, int yy, int pp) { x = xx, y = yy, p = pp; }
    node() {}
} vs[maxn];
double getDis(int x1, int y1, int x2, int y2) {
    return sqrt((double)(x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}
double d[maxn];
bool used[maxn];
double maxD[maxn][maxn];   //MST中从i->j的最大权值
int pre[maxn];          //某一点父节点
bool mst[maxn][maxn];   //该点是否已经在MST中
typedef pair<int, int> P;
double Prim(int s) {
    fill(d, d + maxn, inf);
    fill(pre, pre + maxn, s);
    ms(maxD, 0); ms(used, 0); ms(mst, 0);
    priority_queue<P, vector<P>, greater<P> > q;
    q.push(P(d[s] = 0, s));
    double res = 0;
    while (!q.empty()) {
        P cur = q.top();
        q.pop();
        int u = cur.second;
        if (used[u])
            continue;
        used[u] = true, res += d[u];
        mst[u][pre[u]] = mst[pre[u]][u] = true; //加入到MST中
        for (int v = 1; v <= N; ++v) {
            if (used[v] && w[u][v] < inf)        //只更新MST中的
                maxD[u][v] = maxD[v][u] = max(maxD[pre[u]][v], d[u]);
            if (w[u][v] < d[v]) {
                d[v] = w[u][v];
                pre[v] = u;                     //更新父节点
                q.push(P(d[v], v));
            }
        }
    }
    return res;
}
int main() {
    //freopen("in.txt", "r", stdin);
    int T, a, b, c;
    scanf("%d", &T);
    while (T--) {
        ms(vs, 0); fill(w[0], w[0] + maxn * maxn, inf);
        scanf("%d", &N);
        for (int i = 1; i <= N; ++i) {
            scanf("%d%d%d", &a, &b, &c);
            vs[i] = node(a, b, c);
        }
        for (int i = 1; i < N; ++i)
            for (int j = i + 1; j <= N; ++j)
                w[i][j] = w[j][i] = getDis(vs[i].x, vs[i].y, vs[j].x, vs[j].y);

        //枚举删边, 找出最大值
        double B = Prim(1), A, ans = -1;
        for (int i = 1; i < N; ++i)
            for (int j = i + 1; j <= N; ++j) {
                A = vs[i].p + vs[j].p;
                //这条边未在MST中使用, 尝试加边并删去生成环中的最长边, 已使用则直接变0
                if (mst[i][j]) {
                    ans = max(ans, A / (B - w[i][j]));
                }
                else {
                    ans = max(ans, A / (B - maxD[i][j]));
                }
            }
        printf("%.2lf\n", ans);
    }

    return 0;
}
```

## Matrix tree theorem

基尔霍夫矩阵的任意一个代数余子式是所有生成树的边权积的和。 $A = D - C$, $D$ 度数矩阵，$C$ 邻接矩阵。$z$ 全为 1 的时候就是生成树计数。

```cpp
#include <cstdio>
#include <iostream>
#define LL long long
typedef long long ll;
using namespace std;
int n, m, t, x, y, z, ans = 1;
const ll N = 305, mod = 1e9 + 7;
ll a[N][N];
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
void work() {
    for (int i = 2, inv, tmp; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j)
            if (!a[i][i] && a[j][i]) {
                ans = -ans;
                swap(a[i], a[j]);
                break;
            }
        inv = power(a[i][i], mod - 2, mod);
        for (int j = i + 1; j <= n; ++j) {
            tmp = (LL)a[j][i] * inv % mod;
            for (int k = i; k <= n; ++k)
                a[j][k] = (a[j][k] - (LL)a[i][k] * tmp % mod) % mod;
        }
    }
}
int main() {
    cin >> n >> m >> t;
    for (int i = 1; i <= m; ++i) {
        scanf("%d%d%d", &x, &y, &z);
        if (!t) {
            (a[x][x] += z) %= mod;
            (a[y][y] += z) %= mod;
            (a[x][y] -= z) %= mod;
            (a[y][x] -= z) %= mod;
        } else {
            (a[y][y] += z) %= mod;
            (a[x][y] -= z) %= mod;
        }
    }
    work();
    for (int i = 2; i <= n; ++i) ans = (LL)ans * a[i][i] % mod;
    cout << (ans % mod + mod) % mod;
    return 0;
}
```


```cpp
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <iostream>
using namespace std;
#define mod 10007
int N, R;
struct Point  //点的定义
{
    int x, y;
    Point(int x = 0, int y = 0) : x(x), y(y) {}
} P[301];
Point operator-(Point A, Point B) { return Point(A.x - B.x, A.y - B.y); }
double Cross(Point A, Point B)  //叉积
{
    return A.x * B.y - A.y * B.x;
}
int dcmp(double x)  //精度
{
    if (fabs(x) < 1e-0)
        return 0;
    else
        return x < 0 ? -1 : 1;
}
double Dot(Point A, Point B)  //点积
{
    return A.x * B.x + A.y * B.y;
}
double Distance(Point A, Point B) {
    return (A.x - B.x) * (A.x - B.x) + (A.y - B.y) * (A.y - B.y);
}
bool onSegment(Point p, Point a1, Point a2)  //判断点是否在线段上
{
    return dcmp(Cross(a1 - p, a2 - p)) == 0 && dcmp(Dot(a1 - p, a2 - p)) < 0;
}
bool check(int k1, int k2)  //判断两点之间的距离小于等于R且中间没有点阻隔
{
    if (Distance(P[k1], P[k2]) > R * R) return false;
    for (int i = 0; i < N; i++)
        if (i != k1 && i != k2)
            if (onSegment(P[i], P[k1], P[k2])) return false;
    return true;
}
long long INV(long long a, long long m)  //求a*x=1(mod m)的逆元x
{
    if (a == 1) return 1;
    return INV(m % a, m) * (m - m / a) % m;
}
struct Matrix {
    int mat[301][301];
    Matrix() { memset(mat, 0, sizeof(mat)); }
    int det(int n) {
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                mat[i][j] = (mat[i][j] % mod + mod) % mod;
        int res = 1;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++)
                if (mat[j][i] != 0) {
                    for (int k = i; k < n; k++) swap(mat[i][k], mat[j][k]);
                    if (i != j) res = (mod - res) % mod;
                    break;
                }
            if (mat[i][i] == 0) {
                res = -1;
                break;
            }
            for (int j = i + 1; j < n; j++) {
                int mut = (mat[j][i] * INV(mat[i][i], mod)) % mod;
                for (int k = i; k < n; k++)
                    mat[j][k] =
                        (mat[j][k] - (mat[i][k] * mut) % mod + mod) % mod;
            }
            res = (res * mat[i][i]) % mod;
        }
        return res;
    }
};
int main() {
    int T;
    cin >> T;
    while (T--) {
        int g[301][301];
        memset(g, 0, sizeof(g));
        cin >> N >> R;
        for (int i = 0; i < N; i++) scanf("%d%d", &P[i].x, &P[i].y);
        for (int i = 0; i < N; i++)
            for (int j = i + 1; j < N; j++)
                if (check(i, j)) g[i][j] = g[j][i] = 1;
        Matrix ret;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                if (i != j && g[i][j] == 1) {
                    ret.mat[i][j] = -1;
                    ret.mat[i][i]++;
                }
        cout << ret.det(N - 1) << endl;
    }
    return 0;
}
```

## 匈牙利算法（二分图最大匹配） 

复杂度 $O(n \times e + m)$, n 是左边点数量，m 是右边点数量，e 是图上边数量

```cpp
#include <cstdio>
#include <vector>

const int maxn = 1005;

int n, m, t;
int mch[maxn], vistime[maxn];

std::vector<int> e[maxn];

bool dfs(const int u, const int tag);

int main() {
  scanf("%d %d %d", &n, &m, &t);
  for (int u, v; t; --t) {
    scanf("%d %d", &u, &v);
    e[u].push_back(v);
  }
  int ans = 0;
  for (int i = 1; i <= n; ++i) if (dfs(i, i)) {
    ++ans;
  }
  printf("%d\n", ans);
}

bool dfs(const int u, const int tag) {
  if (vistime[u] == tag) return false;
  vistime[u] = tag;
  for (auto v : e[u]) if ((mch[v] == 0) || dfs(mch[v], tag)) {
    mch[v] = u;
    return true;
  }
  return false;
}
```

## 最小斯坦纳树

```cpp
#include <bits/stdc++.h>
using namespace std;
const int MAXN = 510;
int n, m, k, x, y, z, eg, p[MAXN], hd[MAXN], ver[2 * MAXN], vis[MAXN],
    nx[2 * MAXN], edge[2 * MAXN], dp[MAXN][4200];
priority_queue<pair<int, int> > q;
void add_edge(int x, int y, int z) {
    ver[++eg] = y;
    nx[eg] = hd[x], edge[eg] = z;
    hd[x] = eg;
    return;
}
void dijkstra(int s) {
    memset(vis, 0, sizeof(vis));
    while (!q.empty()) {
        pair<int, int> a = q.top();
        q.pop();
        if (vis[a.second]) {
            continue;
        }
        vis[a.second] = 1;
        for (int i = hd[a.second]; i; i = nx[i]) {
            if (dp[ver[i]][s] > dp[a.second][s] + edge[i]) {
                dp[ver[i]][s] = dp[a.second][s] + edge[i];
                q.push(make_pair(-dp[ver[i]][s], ver[i]));
            }
        }
    }
    return;
}
int main() {
    //freopen("st010.in", "r", stdin);
    //freopen("st010.out", "w", stdout);
    memset(dp, 0x3f, sizeof(dp));
    scanf("%d%d%d", &n, &m, &k);
    for (int i = 1; i <= m; i++) {
        scanf("%d%d%d", &x, &y, &z);
        add_edge(x, y, z), add_edge(y, x, z);
    }
    for (int i = 1; i <= k; i++) {
        scanf("%d", &p[i]);
        dp[p[i]][1 << (i - 1)] = 0;
    }
    for (int s = 1; s < (1 << k); s++) {
        for (int i = 1; i <= n; i++) {
            for (int subs = s & (s - 1); subs; subs = s & (subs - 1)) {
                dp[i][s] = min(dp[i][s], dp[i][subs] + dp[i][s ^ subs]);
            }
            if (dp[i][s] != 0x3f3f3f3f) {
                q.push(make_pair(-dp[i][s], i));
            }
        }
        dijkstra(s);
    }
    printf("%d\n", dp[p[1]][(1 << k) - 1]);
    return 0;
}
```

## 最小树形图（朱刘算法）

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 109, M = 10009; // !!! 注意这里 N M 都开得比较小
struct edge {
    int u, v, w;
} e[M];  //用边表存储

int n, m, root, mn[N], fa[N], tp[N], lp[N], tot, ans;
int zl() {
    while (1) {
        for (int i = 1; i <= n; i++) mn[i] = 1e9, fa[i] = tp[i] = lp[i] = 0;

        for (int i = 1, v, w; i <= m; i++)  // Step 1: 贪心找每个点的最小入边
            if (e[i].u != e[i].v && (w = e[i].w) < mn[v = e[i].v])
                mn[v] = w, fa[v] = e[i].u;
        mn[root] = 0;
        for (int u = 1; u <= n; u++) {
            ans += mn[u];
            if (mn[u] == 1e9) return -1;
        }  // Step 2: 如果有点没有入边就返回 -1 

        for (int u = 1, v = 1; u <= n; u++, v = u) {  // Step 3: 找环并记录
            while (v != root && tp[v] != u && !lp[v]) tp[v] = u, v = fa[v]; // 遍历每个节点uu，然后沿着fa一路逆向走，直到根或者前驱是自己的点（路径压缩）或是环上点
            if (v != root && !lp[v]) {
                lp[v] = ++tot;
                for (int k = fa[v]; k != v; k = fa[k]) lp[k] = tot;
            }
        }
        if (!tot) return ans;  // Step 4: 没环已结束
        for (int i = 1; i <= n; i++)
            if (!lp[i]) lp[i] = ++tot;  // Step 5: 记录孤立点为环

        for (int i = 1; i <= m; i++)  // Step 6: 缩点
            e[i].w -= mn[e[i].v], e[i].u = lp[e[i].u], e[i].v = lp[e[i].v];
        n = tot, root = lp[root], tot = 0;  // Step 7: 重置、进入下一次循环
    }
}

int main() {
    scanf("%d%d%d", &n, &m, &root);
    for (int i = 1, u, v, w; i <= m; i++)
        scanf("%d%d%d", &u, &v, &w), e[i] = (edge){u, v, w};
    printf("%d", zl());
    return 0;
}
```

## 网络流

#### Dinic

> 99%的网络流算法，都可以用Dinic去解。卡Dinic的毒瘤出题人，都是*哔*

```cpp
#include <bits/stdc++.h>
#define up(l, r, i) for (int i = l; i <= r; i++)
#define lw(l, r, i) for (int i = l; i >= r; i--)
using namespace std;
const int MAXN = 10000 + 3;
const int MAXM = 100000 + 3;
const int INF = 2147483647;
typedef long long LL;
inline int qread() {
    int w = 1, c, ret;
    while ((c = getchar()) > '9' || c < '0')
        ;
    ret = c - '0';
    while ((c = getchar()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}
int n, m, s, t, t1, t2, t3, mxflow;
int tot = 1, head[MAXN], nxt[MAXM * 2], ver[MAXM * 2], val[MAXM * 2];
void add(int u, int v, int k) {
    ver[++tot] = v, val[tot] = k, nxt[tot] = head[u], head[u] = tot;
    ver[++tot] = u, val[tot] = 0, nxt[tot] = head[v], head[v] = tot;
}
int dis[MAXN], cur[MAXN];
int q[MAXN], front, rear;
bool bfs() {
    memset(dis, 0, sizeof(dis));
    front = rear = 1, q[rear++] = s, dis[s] = 1;
    while (front < rear) {
        int u = q[front++];
        for (int i = head[u]; i; i = nxt[i]) {
            int v = ver[i];
            if (dis[v] || !val[i]) continue;
            q[rear++] = v, dis[v] = dis[u] + 1;
        }
    }
    return dis[t];
}
int dfs(int u, int flow) {
    if (u == t && ((mxflow += flow) || 1)) return flow;
    int used = 0;
    for (int &i = cur[u]; i; i = nxt[i]) {
        int v = ver[i];
        if (dis[v] == dis[u] + 1) {
            int lft = dfs(v, min(flow - used, val[i]));
            if (lft) val[i] -= lft, val[i ^ 1] += lft, used += lft;
            if (used == flow) return flow;
        }
    }
    return used;
}
int main() {
    n = qread(), m = qread(), s = qread(), t = qread();
    up(1, m, i) t1 = qread(), t2 = qread(), t3 = qread(), add(t1, t2, t3);
    while (bfs()) memcpy(cur, head, sizeof(head)), dfs(s, INF);
    cout << mxflow << endl;
    return 0;
}
```

#### Edmond-Karp

bfs 跑增广路，比 FF 好得多

```cpp
#include <bits/stdc++.h>
#define up(l, r, i) for (register int i = l; i <= r; ++i)
#define erg(u) for (int i = head[u], v = ver[i]; i; i = nxt[i], v = ver[i])
const int INF = 2147483647;
const int MAXN = 10000 + 3, MAXM = 100000 + 3;
using namespace std;
bool vis[MAXN];
int n, m, s, t, tot = 1, U, V, K, ans;
int head[MAXN], nxt[MAXM * 2], ver[MAXM * 2], val[MAXM * 2];
inline void add(int u, int v, int k) {  //加边
    ver[++tot] = v, val[tot] = k, nxt[tot] = head[u], head[u] = tot;
    ver[++tot] = u, val[tot] = 0, nxt[tot] = head[v], head[v] = tot;
}
int q[MAXN][4], front, rear;
inline int EK() {  //主过程
    memset(vis, 0, sizeof(vis)), vis[s] = true;
    front = rear = 1, q[rear][0] = s, q[rear++][3] = INF;
    for (int u = q[front][0]; front < rear; u = q[++front][0])
        erg(u) {                          //遍历当前节点
            if (!vis[v] && val[i] > 0) {  //判断可行性
                q[rear][0] = v, q[rear][1] = i, q[rear][2] = front,
                q[rear++][3] = min(val[i], q[front][3]), vis[v] = true;
                if (v == t) goto end;  //到达终点直接跳出
            }
        }
    return false;
end:
    rear--;
    for (int i = rear, flw = q[rear][3]; q[i][2]; i = q[i][2])
        val[q[i][1]] -= flw, val[q[i][1] ^ 1] += flw;
    return q[rear][3];
}
int main() {
    scanf("%d%d%d%d", &n, &m, &s, &t), vis[s] = true;
    up(1, m, i) scanf("%d%d%d", &U, &V, &K), add(U, V, K);
    for (int p; p = EK();) ans += p;
    printf("%d\n", ans);
    return 0;
}
```


#### Ford Fulksonff

dfs 找增广路，被卡爆炸了别怪我（

```cpp
#include <bits/stdc++.h>
#define up(l, r, i) for (register int i = l; i <= r; ++i)
#define ergv(u)                                                               \
    for (std::vector<edge>::iterator p = head[u].begin(); p != head[u].end(); \
         ++p)
#define ergl(u) \
    for (std::list<int>::iterator p = lst[u].begin(); p != lst[u].end(); ++p)
const int INF = 2147483647;
const int MAXN = 10000 + 3, MAXM = 100000 + 3;
using namespace std;
bool vis[MAXN];
int n, m, s, t, tot = 1, U, V, K, ans;
int head[MAXN], nxt[MAXM * 2], ver[MAXM * 2], val[MAXM * 2];
inline void add(int u, int v, int k) {  //加边
    ver[++tot] = v, val[tot] = k, nxt[tot] = head[u], head[u] = tot;
    ver[++tot] = u, val[tot] = 0, nxt[tot] = head[v], head[v] = tot;
}
inline int FF(int u, int flw) {  //主过程
    if (u == t) return flw;
    vis[u] = true;  //到达终点
    for (int i = head[u]; i; i = nxt[i])
        if (val[i] > 0 && !vis[ver[i]]) {
            int lft = FF(ver[i], min(flw, val[i]));
            if (!lft) continue;  //增广路寻找失败
            val[i] -= lft, val[i ^ 1] += lft, vis[u] = false;
            return lft;
        }
    return vis[u] = false;  // vis函数防止出现环
}
int main() {
    scanf("%d%d%d%d", &n, &m, &s, &t), vis[s] = true;
    up(1, m, i) scanf("%d%d%d", &U, &V, &K), add(U, V, K);
    for (int p; p = FF(s, INF);) ans += p;
    printf("%d\n", ans);
    return 0;
}
```

#### ISAP

```cpp
#include <bits/stdc++.h>
#define up(l, r, i) for (register int i = l; i <= r; ++i)
#define lw(l, r, i) for (register int i = l; i >= r; --i)
using namespace std;
const int MAXN = 1200 + 3;
const int MAXM = 120000 + 3;
const int INF = 2147483647;
typedef long long LL;
inline int qread() {
    int w = 1, c, ret;
    while ((c = getchar()) > '9' || c < '0')
        ;
    ret = c - '0';
    while ((c = getchar()) >= '0' && c <= '9') ret = ret * 10 + c - '0';
    return ret * w;
}
int n, m, s, t, t1, t2, t3, mxflow;
int tot = 1, head[MAXN], nxt[MAXM * 2], ver[MAXM * 2], val[MAXM * 2];
inline void add(int u, int v, int k) {
    ver[++tot] = v, val[tot] = k, nxt[tot] = head[u], head[u] = tot;
    ver[++tot] = u, val[tot] = 0, nxt[tot] = head[v], head[v] = tot;
}
int dis[MAXN], cur[MAXN];
int gap[MAXN], q[MAXN], front, rear;
inline void bfs() {
    front = rear = 1, q[rear++] = t, gap[dis[t] = 1]++;
    while (front < rear) {
        int u = q[front++];
        for (int i = head[u]; i; i = nxt[i]) {
            int v = ver[i];
            if (dis[v]) continue;
            q[rear++] = v, gap[dis[v] = dis[u] + 1]++;
        }
    }
}
int dfs(int u, int flow) {
    if (u == t && ((mxflow += flow) || 1)) return flow;
    int used = 0;
    for (int &i = cur[u]; i; i = nxt[i]) {
        int v = ver[i];
        if (dis[v] == dis[u] - 1) {
            int lft = dfs(v, min(flow - used, val[i]));
            if (lft) val[i] -= lft, val[i ^ 1] += lft, used += lft;
            if (used == flow) return flow;
        }
    }
    (--gap[dis[u]]) ? (++gap[++dis[u]]) : dis[s] = n + 1;
    return used;
}
int main() {
    n = qread(), m = qread(), s = qread(), t = qread();
    up(1, m, i) t1 = qread(), t2 = qread(), t3 = qread(), add(t1, t2, t3);
    bfs();
    while (dis[s] <= n) memcpy(cur, head, sizeof(head)), dfs(s, INF);
    cout << mxflow << endl;
    return 0;
}
```

#### HLPP (TODO)

## 拓扑序

### Kahn 算法

```cpp
void top() {
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (in[i] == 0) q.push(i);
    }
    while(!q.empty()) {
        int u = q.top(); q.pop();
        for (auto nxt: edges[u]) {
            if (--in[nxt] == 0) q.push(nxt);
        }
    }
}
```
