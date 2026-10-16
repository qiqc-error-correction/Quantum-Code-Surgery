"""Independent linear algebra audit; reads an exported deformed-code JSON."""
import json
import sys
from pathlib import Path
import numpy as np


def rref(a):
    a = np.array(a, dtype=np.uint8, copy=True)
    pivots = []
    row = 0
    for col in range(a.shape[1]):
        candidates = np.flatnonzero(a[row:, col])
        if not len(candidates):
            continue
        pivot = row + int(candidates[0])
        a[[pivot, row]] = a[[row, pivot]]
        other = np.flatnonzero(a[:, col])
        other = other[other != row]
        a[other] ^= a[row]
        pivots.append(col)
        row += 1
        if row == len(a):
            break
    return a, pivots


def rank(a):
    return len(rref(a)[1])


def kernel(a):
    rr, pivots = rref(a)
    free = [i for i in range(rr.shape[1]) if i not in pivots]
    out = np.zeros((len(free), rr.shape[1]), dtype=np.uint8)
    for j, col in enumerate(free):
        out[j, col] = 1
        out[j, pivots] = rr[:len(pivots), col]
    return out


def product(x, z, signs, selector):
    px, pz = np.zeros(x.shape[1], dtype=np.uint8), np.zeros(x.shape[1], dtype=np.uint8)
    exponent = 0
    for i in np.flatnonzero(selector):
        exponent += 2 * int(signs[i]) + int(np.sum(x[i] & z[i])) + 2 * int(np.sum(pz & x[i]))
        px ^= x[i]
        pz ^= z[i]
    exponent = (exponent - int(np.sum(px & pz))) % 4
    assert exponent % 2 == 0, 'Antihermitian product'
    return px, pz, exponent // 2


def exhaustive_cheeger(b):
    """Enumerate one representative of every complementary cut pair."""
    n=b.shape[1]
    assert n <= 20 and not (b.sum(axis=1)%2).any()
    rowmasks=[sum(1 << int(i) for i in np.flatnonzero(row)) for row in b]
    best_num,best_den,best_mask=10**9,1,0
    for start in range(1,1 << (n-1),32768):
        masks=np.arange(start,min(start+32768,1 << (n-1)),dtype=np.uint32) << 1
        weight=np.bitwise_count(masks)
        denominator=np.minimum(weight,n-weight)
        boundary=np.zeros(len(masks),dtype=np.uint8)
        for rowmask in rowmasks:
            boundary += np.bitwise_count(masks & rowmask)%2
        for den in np.unique(denominator):
            positions=np.flatnonzero(denominator==den)
            idx=int(positions[np.argmin(boundary[positions])])
            num=int(boundary[idx])
            if num*best_den < best_num*int(den):
                best_num,best_den,best_mask=num,int(den),int(masks[idx])
    return dict(h=best_num/best_den,boundary=best_num,denominator=best_den,
                complementary_cut_pairs=(1 << (n-1))-1,
                worst_cut_zero_based=[i for i in range(n) if best_mask >> i & 1])


def audit(path):
    data = json.loads(Path(path).read_text())
    x, z, signs = [np.asarray(data[k], dtype=np.uint8) for k in ('check_x', 'check_z', 'check_signs')]
    m = data['metadata']
    n = len(m['column_layout']['memory'])
    h = np.hstack((x, z))
    gram = (x.astype(np.int64) @ z.T + z.astype(np.int64) @ x.T) % 2
    assert not gram.any(), 'Noncommuting stabilizers'
    dependencies = kernel(h.T)
    for dep in dependencies:
        px, pz, s = product(x,z,signs,dep)
        assert not px.any() and not pz.any() and s == 0, 'Negative identity stabilizer product'
    no_aux = kernel(np.hstack((x[:, n:], z[:, n:])).T)
    mx, mz = (no_aux @ x[:, :n]) % 2, (no_aux @ z[:, :n]) % 2
    jx, jz = [np.asarray(m['logical_basis'][k],dtype=np.uint8) for k in ('jx', 'jz')]
    signatures = np.hstack(((mx @ jz.T) % 2, (mz @ jx.T) % 2))
    tx, tz = [np.asarray(data[k],dtype=np.uint8) for k in ('measurement_x','measurement_z')]
    target_sig = np.concatenate(((tx @ jz.T) % 2, (tz @ jx.T) % 2))
    assert rank(signatures) == 1 and rank(np.vstack((signatures,target_sig))) == 1
    px,pz,s = product(x,z,signs,np.asarray(data['measurement_witness'],dtype=np.uint8))
    assert np.array_equal(px,np.pad(tx,(0,x.shape[1]-n)))
    assert np.array_equal(pz,np.pad(tz,(0,x.shape[1]-n)))
    assert s == data['measurement_sign']
    # For this disjoint ZX target, the Z factor itself survives as a logical.
    # It is an explicit, search-free static-distance upper-bound witness.
    wx=np.zeros(x.shape[1],dtype=np.uint8)
    wz=np.pad(tz,(0,x.shape[1]-n))
    assert not (tx & tz).any()
    assert not ((x @ wz + z @ wx)%2).any()
    assert rank(np.vstack((h,np.concatenate((wx,wz))))) == rank(h)+1
    b,c = [np.asarray(m[k],dtype=np.uint8) for k in ('graph_incidence','cycle_basis')]
    assert not ((c @ b)%2).any() and rank(c) == len(b)-rank(b)
    report = dict(file=str(Path(path).resolve()),check_matrix_shape=x.shape,n=x.shape[1],stabilizer_rank=rank(h),k=x.shape[1]-rank(h),
        commutator_nonzero=int(np.count_nonzero(gram)),signed_dependency_count=len(dependencies),all_dependencies_positive=True,
        sole_readout_rank=rank(signatures),sole_readout_signature=target_sig.tolist(),signed_target_product_verified=True,
        graph_shape=b.shape,graph_rank=rank(b),cycle_shape=c.shape,cycle_rank=rank(c),
        max_row_weight=int((x|z).sum(axis=1).max()),max_column_weight=int((x|z).sum(axis=0).max()),
        cycle_row_weight=int(c.sum(axis=1).max()),cycle_column_weight=int(c.sum(axis=0).max()))
    report.update(distance_upper_bound=int(np.sum(wx|wz)),distance_witness_x=wx.tolist(),
        distance_witness_z=wz.tolist(),distance_witness_commutes=True,distance_witness_nontrivial=True,
        distance_witness_description='Original weight-10 Z factor on memory, identity on all auxiliary qubits')
    report['independent_expansion']=exhaustive_cheeger(b)
    adapter=m.get('adapter')
    if adapter:
        alen=adapter['adapter_size']
        eleft=len(adapter['edge_layout']['left'])
        nl=len(adapter['P_left'])
        r=n # BB90 has 90 memory check rows
        sel=np.zeros(len(x),dtype=np.uint8);sel[r:r+nl]=1
        lx,lz,ls=product(x,z,signs,sel)
        sel[:]=0;sel[r+nl:r+b.shape[1]]=1
        rx,rz,rs=product(x,z,signs,sel)
        adapter_columns=np.arange(n+eleft,n+eleft+alen)
        assert np.array_equal(np.flatnonzero(lx[n:] | lz[n:])+n,adapter_columns)
        assert np.array_equal(np.flatnonzero(rx[n:] | rz[n:])+n,adapter_columns)
        assert np.array_equal(lx[n:],rx[n:]) and np.array_equal(lz[n:],rz[n:])
        bridge_x=np.zeros(x.shape[1],dtype=np.uint8);bridge_x[adapter_columns]=1
        bridge_z=np.zeros_like(bridge_x)
        assert not ((x @ bridge_z + z @ bridge_x)%2).any()
        assert rank(np.vstack((h,np.concatenate((bridge_x,bridge_z))))) == rank(h)+1
        report.update(adapter_size=alen,individual_products_have_all_adapter_qubits=True,
            adapter_only_logical_x_qubits_one_based=(adapter_columns+1).tolist(),
            adapter_only_logical_weight=alen,adapter_only_logical_commutes_and_is_nontrivial=True)
    return report


if __name__=='__main__':
    result=audit(sys.argv[1])
    text=json.dumps(result,indent=2)
    print(text)
    if len(sys.argv)>2:
        Path(sys.argv[2]).write_text(text+'\n')
