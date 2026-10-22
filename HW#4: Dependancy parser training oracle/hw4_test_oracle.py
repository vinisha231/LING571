import random, sys, time
from hw4_oracle import *

def random_projective(n, rng):
    #Random projective tree over words 1..n via recursive head selection.
    heads = {}
    def build(lo, hi, parent):
        if lo > hi: return
        h = rng.randint(lo, hi)
        heads[h] = parent
        build(lo, h - 1, h)
        build(h + 1, hi, h)
    build(1, n, 0)
    labels = ["det", "nmod", "obj", "nsubj", "case", "amod"]
    return [Token(i, "w%d" % i, "ROOT" if heads[i] == 0 else rng.choice(labels), heads[i])
            for i in range(1, n + 1)]

rng = random.Random(571)
t0 = time.time()
for trial in range(20000):
    s = random_projective(rng.randint(1, 40), rng)
    trans, arcs, ok = run_oracle(s)
    assert ok, s
    got = sorted((a.dep, a.head, a.label) for a in arcs)
    want = sorted((t.index, t.head, t.label) for t in s)
    assert got == want, (s, got)
    assert len(trans) == 2 * len(s) #n SHIFTs + n arcs
print("20000 random projective trees: all round-trip exactly (%.1fs)" % (time.time() - t0))

bk = [Token(1,"book","ROOT",0), Token(2,"the","det",3), Token(3,"flight","obj",1),
      Token(4,"through","case",5), Token(5,"houston","nmod",3)]
trans, arcs, ok = run_oracle(bk)
print("textbook:", [format_transition(t) for t in trans])
assert [format_transition(t) for t in trans] == ["SHIFT","SHIFT","SHIFT","(LEFTARC,det)","SHIFT","SHIFT",
    "(LEFTARC,case)","(RIGHTARC,nmod)","(RIGHTARC,obj)","(RIGHTARC,ROOT)"]

np_ = [Token(1,"a","ROOT",0), Token(2,"b","x",4), Token(3,"c","y",1), Token(4,"d","z",3)]
trans, arcs, ok = run_oracle(np_)
print("non-projective handled gracefully, ok =", ok)
assert ok is False

big = random_projective(5000, rng)
t0 = time.time(); run_oracle(big); print("5000-word sentence: %.3fs" % (time.time() - t0))
t0 = time.time()
for _ in range(2500): run_oracle(random_projective(40, rng))
print("100k tokens: %.2fs" % (time.time() - t0))
print("ALL TESTS PASSED")