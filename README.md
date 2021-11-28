
# 常用头部处理

## 直接上 Template

```cpp
#include <bits/stdc++.h>
using namespace std;

#define rep(i, x, y) for (auto i = (x); i <= (y); i++)
#define dep(i, x, y) for (auto i = (x); i >= (y); i--)
#define DEBUG false
#define ____ puts("\n_______________\n") 
#define debug(x)  if (DEBUG) cout<< #x << " => " << (x) << endl

typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pii;

void init() {
    //
}
void clear() {
    //
}
void solve() {
    //
}
int main() {
    ios::sync_with_stdio(false), cin.tie(0), cout.tie(0);
    int T;
    cin >> T;
    rep(t, 1, T) {
        init();
        solve();
        clear();
    }
    
    return 0;
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

## 树上欧拉序

dfs 时候进栈出栈都记录一次，得到欧拉序列。

### $O(1)$判祖孙关系

```cpp
#include <bits/stdc++.h>
using namespace std;

#define rep(i, x, y) for (auto i = (x); i <= (y); i++)
#define dep(i, x, y) for (auto i = (x); i >= (y); i--)
#define DEBUG false
#define ____ puts("\n_______________\n")
#define debug(x) \
    if (DEBUG) cout << #x << " => " << (x) << endl

typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pii;

const int N = 1e5 + 10;
int n, m;
int head[N], to[N], nxt[N], cnt;
int root;
void add(int x, int y) {  //链式前向星
    to[++cnt] = y;
    nxt[cnt] = head[x];
    head[x] = cnt;
}
int ein[N], eout[N], tot;
void dfs(int x) {
    ein[x] = ++tot;
    for (int i = head[x]; i; i = nxt[i]) {  // 遍历
        int y = to[i];
        if (!ein[y]) dfs(y);
    }
    eout[x] = ++tot;
}
void init() {
    //
}
void clear() {
    //
}
void solve() {
    //
}
bool up(int x, int y) { return (ein[x] <= ein[y] && eout[x] >= eout[y]); }
int main() {
    ios::sync_with_stdio(false), cin.tie(0), cout.tie(0);
    cin >> n;
    int x, y;
    rep(i, 1, n) {
        cin >> x >> y;
        if (y != -1) {
            add(x, y), add(y, x);
        } else {
            root = x;
        }
    }
    dfs(root);
    cin >> m;
    while (m--) {
        cin >> x >> y;
        if (up(x, y)) {
            cout << 1 << endl;
        } else if (up(y, x)) {
            cout << 2 << endl;
        } else {
            cout << 0 << endl;
        }
    }

    return 0;
}
```



## 线段树 (segment tree)

### 数组区间和

```cpp
struct sgt {
    ll ans[MAXN << 2], lazy[MAXN << 2], a[MAXN];
    inline ll ls(ll x) { return x << 1; }
    inline ll rs(ll x) { return x << 1 | 1; }

    inline void push_up(ll u) { ans[u] = ans[ls(u)] + ans[rs(u)]; }
    void build(ll u, ll l, ll r) {
        lazy[u] = 0;
        if (l == r) {
            ans[u] = a[l];
            return;
        }
        ll mid = (l + r) >> 1;
        build(ls(u), l, mid);
        build(rs(u), mid + 1, r);
        push_up(u);
    }
    inline void f(ll u, ll l, ll r, ll k) {
        lazy[u] = lazy[u] + k;
        ans[u] = ans[u] + k * (r - l + 1);
    }
    inline void push_down(ll u, ll l, ll r) {
        ll mid = (l + r) >> 1;
        f(ls(u), l, mid, lazy[u]);
        f(rs(u), mid + 1, r, lazy[u]);
        lazy[u] = 0;
    }
    inline void update(ll nl, ll nr, ll l, ll r, ll u, ll k) {
        if (nl <= l && r <= nr) {
            ans[u] += k * (r - l + 1);
            lazy[u] += k;
            return;
        }
        push_down(u, l, r);
        ll mid = (l + r) >> 1;
        if (nl <= mid) update(nl, nr, l, mid, ls(u), k);
        if (nr > mid) update(nl, nr, mid + 1, r, rs(u), k);
        push_up(u);
    }
    ll query(ll q_x, ll q_y, ll l, ll r, ll u) {
        ll res = 0;
        if (q_x <= l && r <= q_y) return ans[u];
        ll mid = (l + r) >> 1;
        push_down(u, l, r);
        if (q_x <= mid) res += query(q_x, q_y, l, mid, ls(u));
        if (q_y > mid) res += query(q_x, q_y, mid + 1, r, rs(u));
        return res;
    }
} sg;
```

### 数组区间最大值

```cpp
const ll maxn = 3e5; // TODO: change maxn
const ll inf = 1e20;  
ll  la[maxn];        // 用来初始化的数组
ll max1[maxn * 4];
void pushup(ll id) { max1[id] = max(max1[id << 1], max1[id << 1 | 1]); }
void build(ll id, ll l, ll r) {
    if (l == r) {
        max1[id] = la[l];
        return;
    }
    ll mid = (l + r) >> 1;
    build(id << 1, l, mid);
    build(id << 1 | 1, mid + 1, r);
    pushup(id);
}
void update(ll id, ll l, ll r, ll x, ll v) {
    if (l == r) {
        max1[id] = v;
        return;
    }
    ll mid = (l + r) >> 1;
    if (x <= mid)
        update(id << 1, l, mid, x, v);
    else
        update(id << 1 | 1, mid + 1, r, x, v);
    pushup(id);
}
ll query(ll id, ll l, ll r, ll x, ll y) {
    if (x <= l && y >= r) return max1[id];
    ll mid = (l + r) >> 1, ret = -inf;
    if (x <= mid) ret = max(ret, query(id << 1, l, mid, x, y));
    if (y > mid) ret = max(ret, query(id << 1 | 1, mid + 1, r, x, y));
    return ret;
}
```


### 树写法

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

### 线段树维护矩阵

```cpp
#include <bits/stdc++.h>
using namespace std;

#define rep(i, x, y) for (auto i = (x); i <= (y); i++)
#define dep(i, x, y) for (auto i = (x); i >= (y); i--)
#define DEBUG false
#define ____ puts("\n_______________\n")
#define debug(x) \
    if (DEBUG) cout << #x << " => " << (x) << endl

typedef long long ll;
typedef unsigned long long ull;
typedef pair<int, int> pii;
ll MOD = 571373;

void init() {
    //
}
void solve() {
    //
}
void clear() {
    //
}
struct matrix {
    ll a[2][2];
    void crt() {
        a[0][0] = a[1][1] = 1;
        a[0][1] = a[1][0] = 0;
        return;
    }
    void clear() {
        a[0][0] = a[0][1] = a[1][0] = a[1][1] = 0;
        return;
    }
    void crt1(ll x) {
        clear();
        a[0][0] = x;
        a[1][1] = 1;
        return;
    }
    void crt2(ll x) {
        clear();
        a[0][0] = 1;
        a[1][0] = x;
        a[1][1] = 1;
        return;
    }
};

struct node {
    int s, e, m;
    matrix tag;
    bool flag = false;
    ll w[2];

    node *l, *r;

    node(int S, int E) {
        tag.crt();
        s = S, e = E, m = (s + e) / 2;
        if (s != e) {
            l = new node(s, m);
            r = new node(m + 1, e);
        } else {
            w[1] = 1;
        }
    }
    void timesA(matrix y) {
        int f[2];
        f[0] = (w[0] * y.a[0][0] + w[1] * y.a[1][0]) % MOD;
        f[1] = (w[0] * y.a[0][1] + w[1] * y.a[1][1]) % MOD;
        w[0] = f[0], w[1] = f[1];
    }
    void timesB(matrix y) {
        matrix z;
        z.a[0][0] = (tag.a[0][0] * y.a[0][0] + tag.a[0][1] * y.a[1][0]) % MOD;
        z.a[0][1] = (tag.a[0][0] * y.a[0][1] + tag.a[0][1] * y.a[1][1]) % MOD;
        z.a[1][0] = (tag.a[1][0] * y.a[0][0] + tag.a[1][1] * y.a[1][0]) % MOD;
        z.a[1][1] = (tag.a[1][0] * y.a[0][1] + tag.a[1][1] * y.a[1][1]) % MOD;
        tag = z;
        return;
    }
    void apply(matrix ntag) {
        flag = true;
        timesB(ntag);
        timesA(ntag);
    }
    void push() {
        if (s == e) return;
        if (!flag) return;
        flag = false;
        l->apply(tag);
        r->apply(tag);
        tag.crt();
    }
    void update(int S, int E, matrix type) {
        push();
        if (S <= s && e <= E) {
            apply(type);
            return;
        }
        if (S <= m) l->update(S, E, type);
        if (E > m) r->update(S, E, type);
        w[0] = (l->w[0] + r->w[0]) % MOD;
        w[1] = (l->w[1] + r->w[1]) % MOD;
        return;
    }
    long long query(int S, int E) {
        if (S <= s && e <= E) {
            return w[0];
        }
        push();
        ll res = 0;
        if (S <= m) res += l->query(S, E);
        if (E > m) res += r->query(S, E);
        res %= MOD;
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false), cin.tie(0), cout.tie(0);
    int n, m;
    cin >> n >> m >> MOD;
    node root(1, n);
    int tmp;
    rep(i, 1, n) {
        cin >> tmp;
        matrix type;
        type.crt2(tmp);
        root.update(i, i, type);
    }
    int op, x, y, k;
    while (m--) {
        cin >> op;
        if (op == 1 || op == 2) {
            cin >> x >> y >> k;
            matrix type;
            if (op == 1)
                type.crt1(k);
            else
                type.crt2(k);
            root.update(x, y, type);
        } else if (op == 3) {
            cin >> x >> y;
            cout << root.query(x, y) << endl;
        }
    }

    return 0;
}
```

## 树状数组 (BIT)

### 区间查询单点修改

```cpp
#define lowbit(x) ((x) & -(x))
ll arr[N]; void add(ll x, ll k){ while(x <= n){ arr[x] += k; x += lowbit(x); } }
ll qry(ll x){ll sum = 0; while(x) {sum += arr[x]; x -= lowbit(x);} return sum;}
ll qry(ll l , ll r){return qry(r) - qry(l - 1);}
```

### 区间查询区间修改

```cpp
struct bit {
    ll t1[maxn], t2[maxn], n; // !!!: 注意要初始化 n 为你想要设置的数量上限
    
    void _add(ll k, ll v) {
      ll v1 = k * v;
      while (k <= n) {
        t1[k] += v, t2[k] += v1;
        k += k & -k;
      }
    }
    ll _getsum(ll *t, ll k) {
      ll ret = 0;
      while (k) {
        ret += t[k];
        k -= k & -k;
      }
      return ret;
    }
    void add(ll l, ll r, ll v) {
      _add(l, v), _add(r + 1, -v);  // 将区间加差分为两个前缀加
    }
    long long getsum(ll l, ll r) {
      return (r + 1ll) * _getsum(t1, r) - 1ll * l * _getsum(t1, l - 1) -
             (_getsum(t2, r) - _getsum(t2, l - 1));
    }
};
```

### 区间最值单点修改

```cpp
int n;
int c[maxn], d[maxn];  //另开一个数组维护原始成绩值，利用它更新max
void update(int x) {
    while (x <= n) {
        d[x] = c[x];
        int lx = lowbit(x);
        for (int i = 1; i < lx; i <<= 1)  //这里是注意点
            d[x] = max(d[x], d[x - i]);
        x += lowbit(x);
    }
}
int getmax(int l, int r) {
    int ans = 0;
    while (r >= l) {
        ans = max(ans, c[r--]);
        while (r - lowbit(r) >= l) {
            ans = max(ans, d[r]);
            r -= lowbit(r);
        }
    }
    return ans;
}
```

## Treap

### FHQ Treap

```cpp
#include <bits/stdc++.h>
using namespace std;
const int MAX = 1e5 + 1;
int n, m, tot, rt;
struct Treap {
    int pos[MAX], siz[MAX], w[MAX];
    int son[MAX][2];
    bool fl[MAX];
    void pus(int x) { siz[x] = siz[son[x][0]] + siz[son[x][1]] + 1; }
    int build(int x) {
        w[++tot] = x, siz[tot] = 1, pos[tot] = rand();
        return tot;
    }
    void down(int x) {
        swap(son[x][0], son[x][1]);
        if (son[x][0]) fl[son[x][0]] ^= 1;
        if (son[x][1]) fl[son[x][1]] ^= 1;
        fl[x] = 0;
    }
    int merge(int x, int y) {
        if (!x || !y) return x + y;
        if (pos[x] < pos[y]) {
            // if (fl[x]) down(x);
            son[x][1] = merge(son[x][1], y);
            pus(x);
            return x;
        }
        // if (fl[y]) down(y);
        son[y][0] = merge(x, son[y][0]);
        pus(y);
        return y;
    }
    void split(int i, int k, int &x, int &y) {
        if (!i) {
            x = y = 0;
            return;
        }
        // if (fl[i]) down(i);
        if (w[i] <= k)
            x = i, split(son[i][1], k, son[i][1], y);
        else
            y = i, split(son[i][0], k, x, son[i][0]);
        pus(i);
    }
    int kth(int now, int k) {
        while (1) {
            if (k <= siz[son[now][0]])
                now = son[now][0];
            else if (k == siz[son[now][0]] + 1)
                return now;
            else
                k -= siz[son[now][0]] + 1, now = son[now][1];
        }
    }
    void coutt(int i) {
        if (!i) return;
        // if (fl[i]) down(i);
        coutt(son[i][0]);
        printf("%d ", w[i]);
        coutt(son[i][1]);
    }
} Tree;

int main() {
    scanf("%d", &m);
    int op, a, x;
    int y, z;
    int root = 0;
    // for (int i = 1; i <= n; i++) rt = Tree.merge(rt, Tree.build(i));
    for (int i = 1; i <= m; i++) {
        scanf("%d%d", &op, &a);
        if (op == 1) {
            Tree.split(root, a, x, y);
            root = Tree.merge(Tree.merge(x, Tree.build(a)), y);
        } else if (op == 2) {
            Tree.split(root, a, x, z);
            Tree.split(x, a - 1, x, y);
            y = Tree.merge(Tree.son[y][0], Tree.son[y][1]);
            root = Tree.merge(Tree.merge(x, y), z);
        } else if (op == 3) {
            Tree.split(root, a - 1, x, y);
            printf("%d\n", Tree.siz[x] + 1);
            root = Tree.merge(x, y);
        } else if (op == 4) {
            printf("%d\n", Tree.w[Tree.kth(root, a)]);
        } else if (op == 5) {
            Tree.split(root, a - 1, x, y);
            printf("%d\n", Tree.w[Tree.kth(x, Tree.siz[x])]);
            root = Tree.merge(x, y);
        } else if (op == 6) {
            Tree.split(root, a, x, y);
            printf("%d\n", Tree.w[Tree.kth(y, 1)]);
            root = Tree.merge(x, y);
        }
    }
    return 0;
}
```

## 旋转 Treap

```cpp
#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <queue>
typedef long long LL;
using namespace std;
int RD() {
    int out = 0, flag = 1;
    char c = getchar();
    while (c < '0' || c > '9') {
        if (c == '-') flag = -1;
        c = getchar();
    }
    while (c >= '0' && c <= '9') {
        out = out * 10 + c - '0';
        c = getchar();
    }
    return flag * out;
}
//第一次打treap，不压行写注释XD
const int maxn = 1000019, INF = 1e9;
//平衡树，利用BST性质查询和修改，利用随机和堆优先级来保持平衡，把树的深度控制在log
// N，保证了操作效率 基本平衡树有以下几个比较重要的函数：新建，插入，删除，旋转
//节点的基本属性有val(值)，dat(随机出来的优先级)
//通过增加属性，结合BST的性质可以达到一些效果，如siz(子树大小，查询排名)，cnt(每个节点包含的副本数)等
int na;
int ch[maxn][2];  //[i][0]代表i左儿子，[i][1]代表i右儿子
int val[maxn], dat[maxn];
int siz[maxn], cnt[maxn];
int tot, root;
int New(int v) {        //新增节点，
    val[++tot] = v;     //节点赋值
    dat[tot] = rand();  //随机优先级
    siz[tot] = 1;       //目前是新建叶子节点，所以子树大小为1
    cnt[tot] = 1;       //新建节点同理副本数为1
    return tot;
}
void pushup(int id) {  //和线段树的pushup更新一样
    siz[id] = siz[ch[id][0]] + siz[ch[id][1]] +
              cnt[id];  //本节点子树大小 = 左儿子子树大小 + 右儿子子树大小 +
                        //本节点副本数
}
void build() {
    root = New(-INF),
    ch[root][1] = New(INF);  //先加入正无穷和负无穷，便于之后操作(貌似不加也行)
    pushup(root);  //因为INF > -INF,所以是右子树，
}
void Rotate(int &id,
            int d) {  // id是引用传递，d(irection)为旋转方向，0为左旋，1为右旋
    int temp =
        ch[id][d ^ 1];  //旋转理解：找个动图看一看就好(或参见其他OIer的blog)
    ch[id][d ^ 1] =
        ch[temp][d];  //这里讲一个记忆技巧，这些数据都是被记录后马上修改
    ch[temp][d] = id;  //所以像“Z”一样
    id = temp;  //比如这个id，在上一行才被记录过，ch[temp][d]、ch[id][d ^
                // 1]也是一样的
    pushup(ch[id][d]),
        pushup(
            id);  //旋转以后siz会改变，看图就会发现只更新自己和转上来的点，pushup一下,注意先子节点再父节点
}  //旋转实质是({在满足BST的性质的基础上比较优先级}通过交换本节点和其某个叶子节点)把链叉开成二叉形状(从而控制深度)，可以看图理解一下
void insert(int &id, int v) {  // id依然是引用，在新建节点时可以体现
    if (!id) {
        id = New(v);  //若节点为空，则新建一个节点
        return;
    }
    if (v == val[id])
        cnt[id]++;  //若节点已存在，则副本数++;
    else {  //要满足BST性质，小于插到左边，大于插到右边
        int d =
            v < val[id]
                ? 0
                : 1;  //这个d是方向的意思，按照BST的性质，小于本节点则向左，大于向右
        insert(ch[id][d], v);  //递归实现
        if (dat[id] < dat[ch[id][d]])
            Rotate(id, d ^ 1);  //(参考一下图)与左节点交换右旋，与右节点交换左旋
    }
    pushup(id);  //现在更新一下本节点的信息
}
void Remove(int &id, int v) {  //最难de部分了
    if (!id) return;  //到这了发现查不到这个节点，该点不存在，直接返回
    if (v == val[id]) {  //检索到了这个值
        if (cnt[id] > 1) {
            cnt[id]--, pushup(id);
            return;
        }  //若副本不止一个，减去一个就好
        if (ch[id][0] ||
            ch[id]
              [1]) {  //发现只有一个值，且有儿子节点,我们只能把值旋转到底部删除
            if (!ch[id][1] ||
                dat[ch[id][0]] >
                    dat[ch[id]
                          [1]]) {  //当前点被移走之后，会有一个新的点补上来(左儿子或右儿子)，按照优先级，优先级大的补上来
                Rotate(id, 1),
                    Remove(
                        ch[id][1],
                        v);  //我们会发现，右旋是与左儿子交换，当前点变成右节点；左旋则是与右儿子交换，当前点变为左节点
            } else
                Rotate(id, 0), Remove(ch[id][0], v);
            pushup(id);
        } else
            id = 0;  //发现本节点是叶子节点，直接删除
        return;      //这个return对应的是检索到值de所有情况
    }
    v < val[id] ? Remove(ch[id][0], v) : Remove(ch[id][1], v);  //继续BST性质
    pushup(id);
}
int get_rank(int id, int v) {
    if (!id)
        return 0;  //若查询值不存在，返回；因为最后要减一排除哨兵节点，想要结果为-1这里就返回0
    if (v == val[id])
        return siz[ch[id][0]] +
               1;  //查询到该值，由BST性质可知：该点左边值都比该点的值(查询值)小，故rank为左儿子大小
                   //+ 1
    else if (v < val[id])
        return get_rank(ch[id][0],
                        v);  //发现需查询的点在该点左边，往左边递归查询
    else
        return siz[ch[id][0]] + cnt[id] +
               get_rank(
                   ch[id][1],
                   v);  //若查询值大于该点值。说明询问点在当前点的右侧，且此点的值都小于查询值，所以要加上cnt[id]
}
int get_val(int id, int rank) {
    if (!id) return INF;  //一直向右找找不到，说明是正无穷
    if (rank <= siz[ch[id][0]])
        return get_val(
            ch[id][0],
            rank);  //左边排名已经大于rank了，说明rank对应的值在左儿子那里
    else if (rank <= siz[ch[id][0]] + cnt[id])
        return val
            [id];  //上一步排除了在左区间的情况，若是rank在左与中(目前节点)中，则直接返回目前节点(中区间)的值
    else
        return get_val(
            ch[id][1],
            rank - siz[ch[id][0]] -
                cnt[id]);  //剩下只能在右区间找了，rank减去左区间大小和中区间，继续递归
}
int get_pre(int v) {
    int id = root, pre;  //递归不好返回，以循环求解
    while (id) {         //查到节点不存在为止
        if (val[id] < v)
            pre = val[id],
            id = ch[id][1];  //满足当前节点比目标小，往当前节点的右侧寻找最优值
        else
            id = ch
                [id]
                [0];  //无论是比目标节点大还是等于目标节点，都不满足前驱条件，应往更小处靠近
    }
    return pre;
}
int get_next(int v) {
    int id = root, next;
    while (id) {
        if (val[id] > v)
            next = val[id],
            id = ch[id][0];  //同理，满足条件向左寻找更小解(也就是最优解)
        else
            id = ch[id][1];  //与上方同理
    }
    return next;
}
int main() {
    build();  //不要忘记初始化[运行build()会连同root一并初始化，所以很重要]
    na = RD();
    for (int i = 1; i <= na; i++) {
        int cmd = RD(), x = RD();
        if (cmd == 1)
            insert(
                root,
                x);  //函数都写好了，注意：需要递归的函数都从根开始，不需要递归的函数直接查询
        else if (cmd == 2)
            Remove(root, x);
        else if (cmd == 3)
            printf(
                "%d\n",
                get_rank(root, x) -
                    1);  //注意：因为初始化时插入了INF和-INF,所以查询排名时要减1(-INF不是第一小，是“第零小”)
        else if (cmd == 4)
            printf("%d\n",
                   get_val(root, x + 1));  //同理，用排名查询值得时候要查x +
                                           // 1名，因为第一名(其实不是)是-INF
        else if (cmd == 5)
            printf("%d\n", get_pre(x));
        else if (cmd == 6)
            printf("%d\n", get_next(x));
    }
    return 0;
}
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

## 求树的重心

```cpp
inline void getzx(int t, int fat) {
    int i, j;
    sz[t] = 1;
    maxp[t] = 0;
    for (i = head[t]; i; i = e[i].next) {
        j = e[i].to;
        if (j == fat || vis[j]) continue;
        getzx(j, t);
        sz[t] += sz[j];
        maxp[t] = max(sz[j], maxp[t]);
    }
    maxp[t] = max(maxp[t], tot - sz[t]);
    if (maxp[t] < maxp[rt]) rt = t;
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



## 左偏树（可并堆）

```cpp
#include <cstdio>
#include <iostream>

#define MAXN 150010
#define swap my_swap
#define ls S[x].Son[0]
#define rs S[x].Son[1]

using namespace std;
struct Tree {
    int dis, val, Son[2], rt;
} S[MAXN];
int N, T, A, B, C, i;

inline int Merge(int x, int y);
int my_swap(int &x, int &y) { x ^= y ^= x ^= y; }
inline int Get(int x) { return S[x].rt == x ? x : S[x].rt = Get(S[x].rt); }
inline void Pop(int x) {
    S[x].val = -1, S[ls].rt = ls, S[rs].rt = rs, S[x].rt = Merge(ls, rs);
}
inline int Merge(int x, int y) {
    if (!x || !y) return x + y;
    if (S[x].val > S[y].val || (S[x].val == S[y].val && x > y)) swap(x, y);
    rs = Merge(rs, y);
    if (S[ls].dis < S[rs].dis) swap(ls, rs);
    S[ls].rt = S[rs].rt = S[x].rt = x, S[x].dis = S[rs].dis + 1;
    return x;
}
int main() {
    cin >> N >> T;
    S[0].dis = -1;
    for (i = 1; i <= N; ++i) S[i].rt = i, scanf("%d", &S[i].val);
    for (i = 1; i <= T; ++i) {
        scanf("%d%d", &A, &B);
        if (A == 1) {
            scanf("%d", &C);
            if (S[B].val == -1 || S[C].val == -1) continue;
            int f1 = Get(B), f2 = Get(C);
            if (f1 != f2) S[f1].rt = S[f2].rt = Merge(f1, f2);
        } else {
            if (S[B].val == -1)
                printf("-1\n");
            else
                printf("%d\n", S[Get(B)].val), Pop(Get(B));
        }
    }
    return 0;
}
```

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

# DP

## 01 背包

```cpp
#include <iostream>
#include<cstring>
using namespace std;
const int nmax=1000;
 
int v[nmax];//v[i]表示第i个物品的价值value 
int w[nmax];//w[i]表示第i个物品的重量weight 
int dp[nmax];//总价值 
int n,m;//n表示物品数量，m表示背包容量
 
int main(int argc, char** argv) {//一维数组实现的01背包模板 
	while(cin>>n>>m){
		memset(dp,0,sizeof(dp));
		for(int i=0;i<n;i++){
			cin>>w[i]>>v[i];
		}
		for(int i=0;i<n;i++){//遍历n个物品 
			for(int j=m;j>=0;j--){//01背包容量 逆序遍历
			  if(j>=w[i]){
			  	dp[j]=max(dp[j],(dp[j-w[i]]+v[i]));
			  }//第i个物体不选，dp[j]=dp[j];
			   //第i个物体若选	dp[j]=dp[j-w[i]]+v[i]
			} 
		}
		cout<<dp[m]<<endl;
	}
}
```


## 完全背包

```cpp
#include <iostream>
#include<cstring>
using namespace std;
const int nmax=1000;
 
int v[nmax];//v[i]表示第i个物品的价值value 
int w[nmax];//w[i]表示第i个物品的重量weight 
int dp[nmax];//总价值 
int n,m;//n表示物品数量，m表示背包容量
 
int main(int argc, char** argv) {//一维数组实现的完全背包模板 
	while(cin>>n>>m){
		memset(dp,0,sizeof(dp));
		for(int i=0;i<n;i++){
			cin>>w[i]>>v[i];
		}
		for(int i=0;i<n;i++){//遍历n个物品 
			for(int j=0;j<=m;j++){//完全背包容量 顺序遍历
			  if(j>=w[i]){
			  	dp[j]=max(dp[j],(dp[j-w[i]]+v[i]));
			  }//第i个物体不选，dp[j]=dp[j];
			   //第i个物体若选	dp[j]=dp[j-w[i]]+v[i]
			} 
		}
		cout<<dp[m]<<endl;
	}
	
	return 0;
}
```

## 状压


### 枚举子集合
```cpp
for (int sub = S; sub; sub = (sub - 1) & S) {
	// sub 为 S 的子集
}
```

## 数位


统计 $[a,b]$ 内每个数字出现次数

```cpp
#include<iostream>
using std::cin;using std::cout;using std::endl;
long long a,b;//a,b为左右区间
long long ten[20],f[20];//ten[i]=10^i;ten[i]表示i-1位数第i-1位每个数字出现几次
//f[i]表示i位数的每一个数字总共出现f[i]次
long long cnta[20],cntb[20];

void work(long long x,long long *cnt){
    long long num[20] = {0};//num[i]中i>=1，i表示位数，用来存x
    num[0] = 0;//用来计数(长度length)
    while(x){//将数字x存入数组
        num[++num[0]] = x % 10;
        x /= 10;
    }
    for(int i = num[0];i >= 1;i--){//从大位往小位处理
    //由于i位的数字不见得一样，所以需要通过i位数字与i-1位的出现个数相乘
    //得到，再加上该位置该数字出现的个数，即次位数字个次位数出现的个数，
    //同时对于每一位的数字之前的数字还有一部分零散的num2需要加
        for(int j = 0;j <= 9;j++)
         cnt[j] += f[i-1] * num[i];
        for(int j = 0;j < num[i];j++)
         cnt[j] += ten[i-1];
    //所谓num2，其实是对于形如ABC的数(B为当前处理数字，A为B之前的数字串，C为B之
    //后的数字串)，而num2就是C，所以对num2的操作就是将存入数组的C提取出来，变整型
        long long num2 = 0;
        for(int j = i-1;j >= 1;j--)
         num2 = num2*10 + num[j];
        cnt[num[i]] += num2+1;
    //减去每一个零导数字，为当前位数最高位为0的数字的个数，因此减去ten[i-1]
        cnt[0] -= ten[i-1];
    }
   
}

int main(){
    cin >> a >> b;
    ten[0]=1;
    for(int i = 1;i <= 13;i++){
        f[i] = f[i-1]*10 + ten[i-1];//每一个数字出现的次数等于10倍上一层加上这一位在总数中出现的次数
        ten[i] = 10*ten[i-1];//ten[i]=10^i的计算
    }
    work(a-1,cnta);//求得a左侧的数中i出现多少次[1,a-1]
    work(b,cntb);//求得b左侧的数中i出现了多少次[a,b]
    for(int i = 0;i <= 9;i++)
        cout << cntb[i] - cnta[i] << " ";//将结果做差得到区间内的数中i出现了多少次
    return 0;
}
```

## LIS 

```cpp
void solve () {
  fill(dp, dp + top, INF);
  for (int i = 0;i < top; i++) {
    *lower_bound(dp, dp + top, t[i]) = t[i];
  }
  printf("%d\n", lower_bound(dp, dp + top, INF) - dp);
}
```

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

## Lyndon 分解

```cpp
#include <cstdio>
#include <cstring>
char s[5000005];
int main() {
    scanf("%s", s + 1);
    int len = strlen(s + 1);
    int i, j, k, ans = 0;
    i = 1;
    while (i <= len) {
        for (j = i, k = i + 1; k <= len && s[k] >= s[j]; ++k)
            if (s[k] > s[j])
                j = i;
            else
                ++j;
        while (i <= j) ans ^= (i + k - j - 1), i += k - j;
    }
    printf("%d", ans);
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

## 求行列式值

```cpp
#include <algorithm>
#include <cstdio>
#include <cstring>
using namespace std;

typedef long long LL;
const int maxn = 210;
LL a[maxn][maxn];
int n;
LL p;

LL det(int n, LL p) {
    LL ans = 1;
    bool flag;

    for (int i = 1; i <= n; i++) {
        if (!a[i][i]) {
            flag = 0;
            for (int j = i + 1; j <= n; j++)
                if (a[j][i]) {
                    flag = 1;
                    for (int k = i; k <= n; k++) swap(a[i][k], a[j][k]);
                    ans = -ans;
                    break;
                }
            if (!flag) return 0;
        }

        for (int j = i + 1; j <= n; j++) {
            while (a[j][i]) {
                LL t = a[i][i] / a[j][i];
                for (int k = i; k <= n; k++) {
                    a[i][k] -= t * a[j][k];
                    a[i][k] %= p;
                    swap(a[i][k], a[j][k]);
                }
                ans = -ans;
            }
        }
        ans *= a[i][i];
        ans %= p;
    }
    return (ans + p) % p;
}

int main() {
    while (scanf("%d%lld", &n, &p) == 2) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                scanf("%lld", &a[i][j]);
                a[i][j] %= p;
            }
        }
        LL ans = det(n, p);
        printf("%lld\n", ans);
    }
    return 0;
}
```

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

## 欧拉函数

$\varphi(n)$，小于等于 $n$ 的、和 $n$ 互质的数的个数。$\varphi(p) = p - 1, \text{p is prime}$。积性函数：$\varphi(a \times b) = \varphi(a) \times \varphi(b), \text{when} \ gcd(a, b) = 1$。

一些有用的性质:
- $\varphi(p^k) = p^k - p^{k-1}, \text{p is prime}$
- 唯一分解定理可以得到 $n = \prod_{i=1}^s p_i^{k_i}$，于是有 $\varphi(n) = n \times \prod_{i=1}^s \frac{p_i - 1}{p_i}$。


```cpp
int euler_phi(int n) {
  int ans = n;
  for (int i = 2; i * i <= n; i++)
    if (n % i == 0) {
      ans = ans / i * (i - 1);
      while (n % i == 0) n /= i;
    }
  if (n > 1) ans = ans / n * (n - 1);
  return ans;
}
```

## （扩展）欧拉定理

$$a^b = \begin{cases}a^{b \mod \varphi(p)}, & gcd(a,p) = 1 \\ a^b, & gcd(a,p) \neq 1, b < \varphi(p)  \ (\mod \ p) \\ a ^{b \mod \varphi(p) + \varphi(p)}, & gcd(a,p) \neq 1, b \ge \varphi(p) \end{cases}$$

```cpp
#include <cstdio>
int a, m, phi = 1;
int bm, flag;

int qPow(int b, int e) {
	int a = 1;
	for (; e; e >>= 1, b = (long long)b * b % m)
		if(e & 1) a = (long long)a * b % m;
	return a;
}
int euler_phi(int n) {
  int ans = n;
  for (int i = 2; i * i <= n; i++)
    if (n % i == 0) {
      ans = ans / i * (i - 1);
      while (n % i == 0) n /= i;
    }
  if (n > 1) ans = ans / n * (n - 1);
  return ans;
}


int main() {
	scanf("%d%d", &a, &m);
	a %= m;
	int mm = m;
    int phi =  euler_phi(m);
	char ch;
	while ((ch = getchar()) < '0' || ch > '9') ;
	while (bm = bm * 10ll + (ch ^ '0'), (ch = getchar()) >= '0' && ch <= '9')
		if (bm >= phi) flag = 1, bm %= phi;
	if (bm >= phi) flag = 1, bm %= phi;
	if (flag) bm += phi;
	printf("%d", qPow(a, bm));
	return 0;
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

利用 Fermat 小定理用快速幂算 $a$ 的逆元，注意要求模数 $p$ 必须是素数。

```cpp
ll x = power(a, p - 2, mod);
```

线性递推求逆元

```cpp
inv[1] = 1;
for (int i = 2; i <= n; ++i) {
  inv[i] = (long long)(p - p / i) * inv[p % i] % p;
}
```

## 高斯消元 

### float 方程组

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 105;
double a[N][N];
double x[N], eps = 1e-6;
int solve(int n, int m) {
    int c = 0;
    int r;
    for (r = 0; r < n && c < m; r++, c++) {
        int maxr = r;
        for (int i = r + 1; i < n; i++) {
            if (abs(a[i][c]) > abs(a[maxr][c])) maxr = i;
        }
        if (maxr != r) swap(a[r], a[maxr]);
        if (fabs(a[r][c]) < eps) {
            r--; // 当前列全部 0，处理当前行的下一列
            continue;
        }
        for (int i = r + 1; i < n; i++) {
            if (fabs(a[i][c]) > eps) {
                double k = a[i][c] / a[r][c];
                for (int j = c; j < m + 1; j++) a[i][j] -= a[r][j] * k;
                a[i][c] = 0;
            }
        }
    }
    for (int i = r; i < m; i++) {
        if (fabs(a[i][c]) > eps) return -1;  //无解
    }
    if (r < m) return m - r;  //返回自由元个数
    for (int i = m - 1; i >= 0; i--) {
        for (int j = i + 1; j < m; j++) a[i][m] -= a[i][j] * x[j];
        x[i] = a[i][m] / a[i][i];
    }
    return 0;  //有唯一解
}
int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) {
            cin >> a[i][j];
        }
    }
    int pan = solve(n, n);
    if (pan != 0) {
        cout << "No Solution";
        return 0;
    }
    for (int i = 0; i < n; i++) {
        printf("%.2lf\n", x[i]);
    }
}
```

## 异或方程组

```cpp
#include <bits/stdc++.h>
using namespace std;
const int N = 105;
int a[N][N];
int x[N];
int solve(int n, int m) {
    int c = 0;
    int r;
    for (r = 0; r < n && c < m; r++, c++) {
        int maxr = r;
        for (int i = r + 1; i < n; i++) {
            if (abs(a[i][c]) > abs(a[maxr][c])) maxr = i;
        }
        if (maxr != r) swap(a[r], a[maxr]);
        if (a[r][c] == 0) {
            r--; 
            continue;
        }
        for (int i = r + 1; i < n; i++) {
            if (a[i][c] != 0) {
                for (int j = c; j < m + 1; j++) a[i][j] ^= a[r][j];
            }
        }
    }
    for (int i = r; i < m; i++) {
        if (a[i][c] != 0) return -1;  //无解
    }
    if (r < m) return m - r;  //返回自由元个数
    for (int i = m - 1; i >= 0; i--) {
        for (int j = i + 1; j < m; j++) x[i] ^= (a[i][j] && x[j]);
        x[i] = a[i][m];
    }
    return 0;  //有唯一解
}
int main() {
    int n;
    cin >> n;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= n; j++) {
            cin >> a[i][j];
        }
    }
    int pan = solve(n, n);
    if (pan != 0) {
        cout << "No Solution";
        return 0;
    }
    //for (int i = 0; i < n; i++) {
        //printf("%.2lf\n", x[i]);
    //}
}
```


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

## 进制转换

借助十进制为媒介转换

```cpp
int c[10000000];
void transform(int f, int t) { // from to
    int e;
    int sum = 0;
    string a;
    cin >> a;
    for (int x = 0; (int)x < a.size(); x++) {
        if (a[x] < 'A') {
            e = pow(f, (int)a.size() - x - 1);
            e *= (a[x] - '0');
            sum += e;
        } else {
            e = pow(f, (int)a.size() - x - 1);
            e *= (a[x] - 'A' + 10);
            sum += e;
        }
    }
    int g = 0;
    while (sum > 0) {
        c[g++] = sum % t;
        sum /= t;
    }
    for (int i = g - 1; i >= 0; i--) {
        if (c[i] >= 10) {
            printf("%c", c[i] + 'A' - 10);
        } else {
            printf("%d", c[i]);
        }
    }
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
ll ans = 0;  // 逆序对数量
const int maxn = 6e5 + 5;
ll a[maxn], t[maxn];
void merge(ll b, ll e) {
    if (e == b) return;
    ll mid = b + ((e - b) >> 1);
    merge(b, mid);
    merge(mid + 1, e);
    ll i = b, j = mid + 1, s = b;
    while (i <= mid && j <= e)
        if (a[i] <= a[j])
            t[s++] = a[i++];
        else
            t[s++] = a[j++], ans += mid - i + 1;
    while (i <= mid) t[s++] = a[i++];
    while (j <= e) t[s++] = a[j++];
    for (ll l = b; l <= e; l++) a[l] = t[l];
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

## 摩尔投票法

$O(n)$ 求众数

```python
def (nums):
    res = cnt = 0
    for num in nums:
        if cnt == 0:
            res = num
            cnt += 1
        elif res == num:
            cnt += 1
        else:
            cnt -= 1
    return res
```

## 离散化

```cpp
int main() {
    scanf("%d", &n);
    for (int i = 1; i <= n; i++) scanf("%d", &a[i]), dis[i] = a[i];
    sort(dis + 1, dis + n + 1);
    int num = unique(dis + 1, dis + n + 1) - dis - 1;
    for (int i = 1; i <= n; i++)
        a[i] = lower_bound(dis + 1, dis + num + 1, a[i]) - dis;
    for (int i = 1; i <= n; i++) printf("%d ", dis[a[i]]);
    return 0;
}

```

## 快速选择

借助快排的思想，$O(logn)$ 选出数列第 k 大（小）的元素

```cpp
#include <bits/stdc++.h>
using namespace std;

int partition(int arr[], int l, int r) {
    int x = arr[r], i = l;
    for (int j = l; j <= r - 1; j++) {
        if (arr[j] <= x) {
            swap(arr[i], arr[j]);
            i++;
        }
    }
    swap(arr[i], arr[r]);
    return i;
}

int kthSmallest(int arr[], int l, int r, int k) {
    if (k > 0 && k <= r - l + 1) {
        int index = partition(arr, l, r);

        if (index - l == k - 1) return arr[index];
        if (index - l > k - 1) return kthSmallest(arr, l, index - 1, k);

        return kthSmallest(arr, index + 1, r, k - index + l - 1);
    }

    return INT_MAX;
}

// Driver program to test above methods
int main() {
    int arr[] = {10, 4, 5, 8, 6, 11, 26};
    int n = sizeof(arr) / sizeof(arr[0]);
    int k = 3;
    cout << "K-th smallest element is " << kthSmallest(arr, 0, n - 1, k);
    return 0;
}
```

# 计算几何

计算几何通用头

**900 行的大模板在 [./geometry](./geometry.md) 里面**，下面的已经被淘汰。

```cpp
namespace geometry {

struct Vector {
    double x, y;
    Vector(double xx = 0, double yy = 0) : x(xx), y(yy) {}
};
typedef Vector Point;
const double epsilon = 1e-10;

Vector operator+(Vector a, Vector b) { return Vector(a.x + b.x, a.y + b.y); }
Vector operator-(Vector a, Vector b) { return Vector(a.x - b.x, a.y - b.y); }
Vector operator*(Vector a, double p) { return Vector(a.x * p, a.y * p); }
Vector operator/(Vector a, double p) { return Vector(a.x / p, a.y / p); }

bool operator<(const Point &a, const Point &b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
}

int dcmp(double x) {
    if (fabs(x) < epsilon)
        return 0;
    else
        return x < 0 ? -1 : 1;
}
bool operator==(const Point &a, const Point &b) {
    return (!dcmp(a.x - b.x)) && (!dcmp(a.y - b.y));
}

double dot(Vector a, Vector b) { return a.x * b.x + a.y * b.y; }
double length(Vector a)  // Vector
{
    return sqrt(dot(a, a));
}
double angle(Vector a, Vector b) {
    return acos(dot(a, b) / length(a) / length(b));
}

double cross(Vector a, Vector b) { return a.x * b.y - a.y * b.x; }
double area_2(Point a, Point b, Point c) { return cross(b - a, c - a); }
double length(Point a, Point b)  // 2 points
{
    return sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

Vector rotate(Vector a, double rad) {
    return Vector(a.x * cos(rad) - a.y * sin(rad),
                  a.x * sin(rad) + a.y * cos(rad));
}
Vector normal(Vector a) {
    double l = length(a);
    return Vector(-a.y / l, a.x / l);
}
}  // namespace geometry
```

## 扫描线 (TODO)

## 凸包

算凸包周长

```cpp
#include <bits/stdc++.h>
using namespace std;
int n;
struct vec {
    double x, y;
    vec(double xx = 0, double yy = 0) : x(xx), y(yy) {}
    vec operator-(const vec &a) { return vec(x - a.x, y - a.y); }
} p[100010], sta[100010];
int top;
double K(vec a) { return atan2(a.y, a.x); }
double dis(vec a, vec b) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}
bool cmp(vec a, vec b) {
    double t = K(a - p[1]) - K(b - p[1]);
    if (t)
        return t < 0;
    else
        return dis(p[1], a) < dis(p[1], b);
}
double cp(vec a, vec b) { return a.x * b.y - a.y * b.x; }
double ans;
int main() {
    scanf("%d", &n);
    if (n == 1) return putchar('0'), 0;
    for (int i = 1; i <= n; ++i) {
        scanf("%lf%lf", &p[i].x, &p[i].y);
        if (p[1].y > p[i].y)
            swap(p[1], p[i]);
        else if (p[1].y == p[i].y && p[1].x > p[i].x)
            swap(p[1], p[i]);
    }
    sort(p + 2, p + n + 1, cmp);
    sta[1] = p[1];
    sta[top = 2] = p[2];
    for (int i = 3; i <= n; ++i) {
        while (top >= 2 &&
               cp(p[i] - sta[top - 1], sta[top] - sta[top - 1]) >= 0)
            --top;
        sta[++top] = p[i];
    }
    sta[++top] = p[1];
    for (int i = 1; i < top; ++i) ans += sqrt(dis(sta[i], sta[i + 1]));
    printf("%.2f", ans);
}
```

## 求直线交点

```cpp
#include<bits/stdc++.h>
#define ll long long
using namespace std;
const double eps = 1e-10;

inline int sign(const double &x)
{
	if(x>eps) return 1;
	if(x<-eps) return -1;
	return 0;
}
struct Point
{
	double x,y;
		Point(double _x=0,double _y=0):x(_x),y(_y) {	}
		Point operator -(const Point &op2) const {
		return Point(x-op2.x,y-op2.y);
	}
		double operator ^(const Point &op2) const {
		return x*op2.y-y*op2.x;
	}
};
inline double sqr(const double &x) {
	return x*x;
}
inline double mul(const Point &p0,const Point &p1,const Point &p2) {
	return (p1-p0) ^ (p2-p0);
}
inline double dis2(const Point &p0,const Point &p1) {
	return sqr(p0.x-p1.x)+sqr(p0.y-p1.y);
}
inline double dis(const Point &p0,const Point &p1) {
	return sqrt(dis2(p0,p1));
}
inline int cross(const Point &p1,const Point &p2,const Point &p3,const Point &p4,Point &p) {
	double a1 = mul(p1,p2,p3),a2 = mul(p1,p2,p4);
	if(sign(a1)==0 && sign(a2)==0) return 2;
	if(sign(a1-a2)==0) return 0;
	p.x = (a2*p3.x-a1*p4.x) /(a2-a1);
	p.y = (a2*p3.y-a1*p4.y) /(a2-a1);
	return 1;
}
Point p1,p2,p3,p4,p;
Point e[1005], ans[1005];
bool cmp(Point a, Point b) {
	return dis(a, p1) < dis(b, p1);
}
int main()
{
	int n, m, id = 0;
	cin >> n >> m;
	cin >> p1.x >> p1.y >> p2.x >> p2.y;
	for (int i = 1; i <= n; i++) {
		cin >> e[i].x >> e[i].y;
	}
	for (int op = 1; op <= m; op++) {
		int h, k;
		id = 0;
		cin >> h >> k;
		p3.x = e[h].x;
		p3.y = e[h].y;
		for (int i = 1; i <= n; i++) {
			if (i == h) continue;
			p4.x = e[i].x;
			p4.y = e[i].y;
			int flag = cross(p1,p2,p3,p4,p);
			if (flag == 0 || flag == 2) continue;
			else {
				if (p.x < p1.x && p.x < p2.x) continue;
				if (p.y < p1.y && p.y < p2.y) continue;
				if (p.x > p1.x && p.x > p2.x) continue;
				if (p.y > p1.y && p.y > p2.y) continue;
				ans[++id].x = p.x;
				ans[id].y = p.y;
			}
		}
		sort(ans+1, ans+1+id, cmp);
		if (id < k) cout << -1 << endl;
		else {
			printf("%.6lf %.6lf\n", ans[k].x, ans[k].y);
		}
	}
	return 0;
}
```

# C++ 相关

## 位运算

- `int __builtin_ffs (unsigned int x)`: 返回x的最后一位1的是从后向前第几位，比如7368（1110011001000）返回4。
- `int __builtin_clz (unsigned int x)`: 返回前导的0的个数。
- `int __builtin_ctz (unsigned int x)`: 返回后面的0个个数，和__builtin_clz相对。
- `int __builtin_popcount (unsigned int x)`: 返回二进制表示中1的个数。
- `int __builtin_parity (unsigned int x)`: 返回x的奇偶校验位，也就是x的1的个数模2的

# STL

## bitset

### Constructor

- `std::bitset<8> b1;`： 默认全 0 
- `std::bitset<8> b2(42);`：`unsigned long long init`
```cpp
template< class CharT, class Traits, class Alloc >
explicit bitset( const std::basic_string<CharT,Traits,Alloc>& str,
                 typename std::basic_string<CharT,Traits,Alloc>::size_type
                     pos = 0,
                 typename std::basic_string<CharT,Traits,Alloc>::size_type
                     n = std::basic_string<CharT,Traits,Alloc>::npos,
                 CharT zero = CharT('0'),
                 CharT one = CharT('1'));
```
- `std::bitset<8> b7("XXXXYYYY", 8, 'X', 'Y')`: string init, custom size, pos and 01 chars


- `constexpr bool operator[]( std::size_t pos ) const;`：选择某位，注意位是从低位到高位的
- `bool all() const noexcept;`: all 1
- `bool any() const noexcept;`: any 1
- `bool none() const noexcept;`: none 1
- `std::size_t count() const noexcept;`: 多少个 1 
- `bool test(std::size_t pos) const;`: 返回 pos 位的值，比 `[]` 多边界检查

### Capacity
- `constexpr std::size_t size() const noexcept;`: 大小，即初始化模板里的参数

### Modifiers
- `bitset& set( std::size_t pos, bool value = true );`: 直接使用 `b.set() `全设为 1，否则按参数处理
- `bitset& reset( std::size_t pos );`: set bit to false
- `bitset& flip( std::size_t pos )`: true to false, false to true, 直接使用全部反转

### Convertions

```cpp
template<
    class CharT = char,
    class Traits = std::char_traits<CharT>,
    class Allocator = std::allocator<CharT>
> std::basic_string<CharT,Traits,Allocator>
    to_string(CharT zero = CharT('0'), CharT one = CharT('1')) const;
```
- to_string: `b.to_string('0', 'X')`
- to_ulong: `unsigned long to_ulong() const`
- to_ullong: `unsigned long long to_ullong() const`
 
## set

### Modifiers
- insert: `std::pair<iterator,bool> insert( value_type&& value );`: 插入
- clear: `void clear() noexcept;`: 清除 set
- erase: `iterator erase( const_iterator pos );`: 删除元素
### Capacity
- empty: `bool empty() const noexcept;`: 是否没有元素
- size: `size_type size() const noexcept;`: 返回元素数量
### Lookup
- find: `const_iterator find( const Key& key ) const;`: 查找元素
- count: `size_type count( const Key& key ) const`: 查找元素数量，只会返回 0 或 1
- 

## priority_queue

- 自定义 cmp 函数
```cpp
struct node {
    int i, j, num, f;
};  
struct cmp1 {
    bool operator()(node x, node y) { return x.num > y.num; }
};                                           
priority_queue<node, vector<node>, cmp1> q;  
```

## vector

- `constexpr void assign( size_type count, const T& value );`: 替换 vector 中元素 

# STL Algorithms

## `std::lower_bound`

```cpp
template< class ForwardIt, class T, class Compare >
constexpr ForwardIt lower_bound( ForwardIt first, ForwardIt last, const T& value, Compare comp );
```

- `[first, last)` 中找第一个**不小于** (not less than) value 的元素。
- `comp`: binary predicate which returns `true` fi the first argument is less than (ordered before) the second
 
## `std::upper_bound`

```cpp
template< class ForwardIt, class T, class Compare >
constexpr ForwardIt upper_bound( ForwardIt first, ForwardIt last, const T& value, Compare comp );
```
- Returns an iterator pointing to the first element in the range [first, last) that is **greater than** value, or last if no such element is found.


## `std::binary_search`

```cpp
template< class ForwardIt, class T >
constexpr bool binary_search( ForwardIt first, ForwardIt last, const T& value );
```

- 二分搜索判断一个值存不存在，返回 1 或者 0。注意需要先 `sort`！

## `std::accumulate`

```cpp
template< class InputIt, class T >
constexpr T accumulate( InputIt first, InputIt last, T init );
```

- 计算 $[first, last)$ 的和




# C++ USK

## sort with lambda

```cpp
sort(a, a + 4, [](int a, int b) { return a > b; });
```


# 注意！

> 十年OI一场空，不开long long见祖宗

一下常见错误：

- 没用 long long 或者 double 数据溢出
- mod 过程中数据溢出、没有用逆元计算、每一步都需要 mod，否则很有可能算法正确但是 mod 中间出错
- 题意读错，漏掉细节，看清楚到底要求什么、题目的一些细小限制
- 忘了重置相关数组和变量


# References

- [ACM-axiomofchoice](https://github.com/axiomofchoice-hjt/ACM-axiomofchoice)
- [南京大学 ACM 队 EC Final Template](https://upload-file.xcpcio.com/Code-Library/NJU-Calabash-Release-For-EC-Final.pdf)
