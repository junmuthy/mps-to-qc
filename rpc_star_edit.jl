using ITensors, ITensorMPS
let
    tT0 = time()
    println("Initializing sites")
    t0 = time()
    N_spatial = 3

    # toggle between different basis representations
    basis = "Electron" # local dimension 4 (|0> |up> |dn> |up dn>) requires a bit extra setup/thinking to get right; can utilize N and S_z conservation
    # basis = "Fermion" # local dimension 2 (|0> |1>) easier to implement; can utilize N conservation, can't invoke S_z conservation

    if basis == "Electron"
      N = N_spatial
      sites = siteinds("Electron", N; conserve_qns=false) # conserve_qns=true is nice when scanning over particle sectors and S_z if relevant, but will not leave those sectors 'state' not initialized for those sectors; may not match expected ground state
      state = [isodd(j) ? "Up" : "Dn" for j in 1:N]
    else
      N = 2*N_spatial
      sites = siteinds("Fermion", N; conserve_qns=false) # conserve_qns=true is nice when scanning over particle sectors and S_z if relevant, but will not leave those sectors 'state' not initialized for those sectors; may not match expected ground state
    end

    U = 1.0
    ed = -0.5      # impurity level energy
    epsilon = 1.0  # bath on-site energy
    V = 0.1        # impurity-bath hybridization
    tf = time()
    println("$(tf-t0) seconds")

    println("Building MPO sum")
    t0 = time()
    os = AutoMPO()

    if basis == "Electron"
      # Impurity term: sum_sigma ed n_{1 sigma} + U n_{1 up} n_{1 dn}
      os += (ed, "Nup", 1)
      os += (ed, "Ndn", 1)
      os += (U, "Nup", 1, "Ndn", 1)

      # Bath + hybridization in star geometry
      for j = 2:N_spatial # go over all baths, assuming j=1 is impurity
        os += (epsilon, "Nup", j)
        os += (epsilon, "Ndn", j)

        os += (V, "Cdagup", 1, "Cup", j)
        os += (V, "Cdagup", j, "Cup", 1)

        os += (V, "Cdagdn", 1, "Cdn", j)
        os += (V, "Cdagdn", j, "Cdn", 1)
      end
    else
      # Impurity term: sum_sigma ed n_{1 sigma} + U n_{1 up} n_{1 dn}
      os += (ed, "N", 1)
      os += (ed, "N", 2)
      os += (U, "N", 1, "N", 2)

      # Bath + hybridization in star geometry.
      # Map each spatial orbital a to fermion sites (2a-1, 2a) = (up, dn).
      for bath = 2:N_spatial
        jup = 2 * bath - 1
        jdn = 2 * bath

        os += (epsilon, "N", jup)
        os += (epsilon, "N", jdn)

        os += (V, "Cdag", 1, "C", jup)
        os += (V, "Cdag", jup, "C", 1)

        os += (V, "Cdag", 2, "C", jdn)
        os += (V, "Cdag", jdn, "C", 2)
      end
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
    if basis == "Electron"
      psi0 = random_mps(sites, state; linkdims=10)
    else
      psi0 = random_mps(sites; linkdims=10)
    end
    tf = time()
    println("$(tf-t0) seconds")

    nsweeps = 5
    maxdim = [10]
    cutoff = [1E-10]

    println("Running DMRG")
    t0 = time()
    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    tf = time()
    println("$(tf-t0) seconds")

    tTf = time()
    println("Total time: $(tTf-tT0) seconds")
    println("Ground state energy = $energy")
    println("Done")
  return
end
