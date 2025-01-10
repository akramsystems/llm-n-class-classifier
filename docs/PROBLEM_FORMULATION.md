## Mathematical Problem Formulation
    
### Given:

Set of inputs `X={x1,x2,…,xn}:`
The input data points that need classification.

Set of user-defined labels `L={l1,l2,…,lk}:`
Custom labels provided by end-users, along with their descriptions
`D={d1,d2,…,dk}.`

Classification tasks:
    Binary classification `(L={0,1})`
    Multi-class classification `(|L|>2)`

Few-shot examples `F={(xf,lf)}f=1m`:
A small set of labeled examples provided for the system to learn from,
where `xf∈X` and `lf∈L`.

### Objective:

Design a system S such that:

`S: X → L`, where `S(x)` predicts a label `l` in `L` for each input `x` in `X`.
