import logging
from z3 import *
from flops import TransformerHparams
from multiprocessing import Pool
from tqdm import tqdm

def reduction_z3_chunk(args):
    S_range, H_range, L_range, V_range, I_range, A_range = args
    solver = Solver()

    S = Int('S')
    solver.add(S % 16 == 0)
    solver.add(S >= S_range[0])
    solver.add(S <= S_range[1])
    H = Int('H')
    solver.add(H >= H_range[0])
    solver.add(H % 16 == 0)
    solver.add(H <= H_range[1])
    L = Int('L')
    solver.add(L >= L_range[0])
    solver.add(L <= L_range[1])
    V = Int('V')
    solver.add(V >= V_range[0])
    solver.add(V % 1000 == 0)
    solver.add(V <= V_range[1])
    I = Int('I')
    solver.add(I >= I_range[0])
    solver.add(I % 32 == 0)
    solver.add(I <= I_range[1])
    A = Int('A')
    solver.add(A >= A_range[0])
    solver.add(A <= A_range[1])
    solver.add(H % A == 0)

    model = TransformerHparams(H, L, S, V, I, A)
    params = model.get_params()

    solver.add(params * 4 / 1e6 <= 3)

    result = []

    while solver.check() == sat:
        model = solver.model()

        s_value = model[S].as_long()
        h_value = model[H].as_long()
        l_value = model[L].as_long()
        v_value = model[V].as_long()
        i_value = model[I].as_long()
        a_value = model[A].as_long()

        result.append((s_value, h_value, l_value, v_value, i_value, a_value))
        solver.add(
            Or(S > s_value, H > h_value, L > l_value, V > v_value, I > i_value, A > a_value)
        )

    return result

def reduction_z3():
    S_range = (256, 512)
    H_range = (16, 768)
    L_range = (1, 12)
    V_range = (1000, 50265)
    I_range = (16, 3072)
    A_range = (1, 12)

    # Divide each range into smaller subranges
    S_chunks = [(i, min(i + 32, S_range[1])) for i in range(S_range[0], S_range[1], 32)]
    H_chunks = [(H_range[0], H_range[1])]
    L_chunks = [(i, min(i + 1, L_range[1])) for i in range(L_range[0], L_range[1], 1)]
    V_chunks = [(i, min(i + 1000, V_range[1])) for i in range(V_range[0], V_range[1], 1000)]
    I_chunks = [(I_range[0], I_range[1])]
    A_chunks = [(i, min(i + 1, A_range[1])) for i in range(A_range[0], A_range[1], 1)]

    # Create a list of tuples for all combinations of subranges
    chunks = [(S, H, L, V, I, A) for S in S_chunks for H in H_chunks for L in L_chunks for V in V_chunks for I in I_chunks for A in A_chunks]

    with Pool(processes=80) as pool, tqdm(total=len(chunks)) as pbar:
        results = []
        for result in pool.imap_unordered(reduction_z3_chunk, chunks):
            results.extend(result)
            pbar.update()

    results = list(set(results))

    S_range = (min([x[0] for x in results]), max([x[0] for x in results]))
    H_range = (min([x[1] for x in results]), max([x[1] for x in results]))
    L_range = (min([x[2] for x in results]), max([x[2] for x in results]))
    V_range = (min([x[3] for x in results]), max([x[3] for x in results]))
    I_range = (min([x[4] for x in results]), max([x[4] for x in results]))
    A_range = (min([x[5] for x in results]), max([x[5] for x in results]))

    lower_bound = [
        1,
        V_range[0],
        L_range[0],
        H_range[0],
        1,
        0.2,
        I_range[0],
        A_range[0],
        0.2,
        S_range[0],
        1,
        1,
        1
    ]
    upper_bound = [
        4,
        V_range[1],
        L_range[1],
        H_range[1],
        4,
        0.5,
        I_range[1],
        A_range[1],
        0.5,
        S_range[1],
        3,
        3,
        2
    ]

    return lower_bound, upper_bound

if __name__ == "__main__":
    lb, ub = reduction_z3()
    print(lb)
    print(ub)
