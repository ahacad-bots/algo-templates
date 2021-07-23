

# 常用头部处理

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

# 树

## 线段树

```cpp
struct segment {
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

## 树链剖分

```cpp

```
