using ITensors, ITensorMPS
let
    tT0 = time()
    println("Initializing sites")
    t0 = time()
    N = 3
    sites = siteinds("Electron", N; conserve_qns=false)
    state = [isodd(j) ? "Up" : "Dn" for j in 1:N] 
    U = 1.0
    epsilon = 1.0
    V = 0.1
    tf = time()
    println("$(tf-t0) seconds")

    println("Building MPO sum")
    t0 = time()
    os = AutoMPO()
    os += (U,"Nup",1,"Ndn",1)
    os -= (U/2,"Nup",1)
    os -= (U/2,"Ndn",1)
    os += (U/4,"I",1)
    for j=2:N
      os += (epsilon,"Nup",j)
      os += (epsilon,"Ndn",j)
      os += (V,"Cdagup",1,"Cup",j)
        os += (V,"Cup",1,"Cdagup",j)
        os += (V,"Cdagdn",1,"Cdn",j)
        os += (V,"Cdn",1,"Cdagdn",j)
    end
    tf = time()
    println("$(tf-t0) seconds")
    
    println("Making Hamiltonian MPO")
    t0 = time()
    H = MPO(os, sites)
    tf = time()
    println("$(tf-t0) seconds")

    println("Initializing MPS")
    t0 = time()
    psi0 = random_mps(sites, state; linkdims=10)
    tf = time()
    println("$(tf-t0) seconds")

    nsweeps = 5
    maxdim = [10]
    cutoff = [1E-10]

    println("Running DMRG")
    t0 = time()
    energy,psi = dmrg(H,psi0;nsweeps,maxdim,cutoff)
    tf = time()
    println("$(tf-t0) seconds")
    tTf = time()
    println("Total time: $(tTf-tT0) seconds")
    println("Ground state energy = $energy")
    println("Done")
  return
end
