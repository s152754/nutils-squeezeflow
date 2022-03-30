import numpy as np
from nutils import cli, mesh, function, solver, export, types, sys
import numpy, treelog, typing, pandas, os, os.path

unit = types.unit(m=1, s=1, g=1e-3, K=1, N='kg*m/s2', Pa='N/m2', J='N*m', W='J/s', bar='0.1MPa', min='60s', hr='60min')
_ = numpy.newaxis
# ### load txt file for monte carlo sampling
# def sample():
#     parameters = numpy.loadtxt('NewtonianParam.txt',skiprows=1)
#
#     treelog.user(parameters)
#     for F,eta,R0,V in parameters:
#         print('v sol',V)
#       #  results = main(muinf=eta,eta,l,nu,R0,h0,F,T,m,urdegree,uzdegree,npicard,tol,nt)
#       #  treelog.user('resultaten sample:',results)

def sample(fname:str, T:unit['s'], surf:unit['mN/m'], m:int, urdegree:int, uzdegree:int, npicard:int, tol:float, nt:int):
    '''
    Radial squeeze flow of a Carreau fluid

    .. arguments::

       fname [TestCarreauMC2.txt]
         File name with Monte Carlo data

       T [60s]
         Final time

       surf [63.4mN/m]
         surface tension

       m [10]
         Number of (radial) elements

       nt [10]
         Number of time steps

       npicard [50]
         Number of Picard iterations

       tol [0.001]
         Tolerance for Picard iterations

       urdegree [3]
         Velocity order in radial direction

       uzdegree [6]
         Velocity order in thickness direction

    .. presets::

       newtonian
        urdegree=3
        uzdegree=6
        m=8
        nt=8
    '''

    # read the complete input file
    indata = numpy.loadtxt(fname, skiprows=1, delimiter=',')

    # check whether a data directory exists
    if not os.path.exists('TestCarreau'):
        os.mkdir('TestCarreau')



    for irun in range(indata.shape[0]):

        # create run file name
        fout = 'TestCarreau/{:06d}.txt'.format(int(indata[irun,0]))

        # check whether a file already exists
        if os.path.exists(fout):
            treelog.user('{} already done.'.format(fout))
            continue

        etainf, eta0, n, l, F, R0, V = indata[irun,1:]

        muinf = etainf
        mu0   = eta0
        nu    = n
        h0    = V/(numpy.pi*R0**2)
        treelog.user(f'mu = {muinf}')

        arr = main(muinf=muinf, mu0=mu0, l=l, nu=nu, R0=R0, h0=h0, F=F, T=T, surf=surf, m=m, urdegree=urdegree, uzdegree=uzdegree, npicard=npicard, tol=tol, nt=nt)

        numpy.savetxt(fout, arr, delimiter=',')

        if irun > 3:
            break




def main(muinf:unit['Pa*s'], mu0:unit['Pa*s'], l:unit['s'], nu:float, R0:unit['m'], h0:unit['m'], F:unit['N'], T:unit['s'], surf:unit['mN/m'], m:int, urdegree:int, uzdegree:int, npicard:int, tol:float, nt:int):

    '''
    Radial squeeze flow of a Carreau fluid

    .. arguments::

       muinf [0.86Pa*s]
         Viscosity at infinite shear rate

       mu0 [0.86Pa*s]
         Viscosity at zero shear rate

       l [0.005s]
         Relaxation time

       nu [2.]
         Power index
       
       R0 [2cm]
         Initial radius of fluid domain

       h0 [1mm]
         Initial height of fluid domain

       F [2.8N]
         Loading

       T [60s]
         Final time

       surf [63.4mN/m]
         surface tension

       m [10]
         Number of (radial) elements

       nt [10]
         Number of time steps

       npicard [50]
         Number of Picard iterations

       tol [0.001]
         Tolerance for Picard iterations

       urdegree [3]
         Velocity order in radial direction

       uzdegree [6]
         Velocity order in thickness direction

    .. presets::

       newtonian
        nu=1.0
        l=0.0s
        surf=0mN/m
        urdegree=3
        uzdegree=2
        m=8
        nt=10

       carreau
        urdegree=3
        uzdegree=6
        m=4
        nt=8
        tol=0.000001
    '''

    Rinit    = R0
    hinit    = h0

    pdegree  = urdegree-1

    ρdomain, ρgeom = mesh.line(numpy.linspace(0, 1, m+1), bnames=('inner','outer'), space='R')
    ζdomain, ζgeom = mesh.line(numpy.linspace(-1/2, 1/2, 2), bnames=('bottom','top'), space='H')

    domain = ρdomain*ζdomain

    ns = function.Namespace()
    ns.hbasis = [1.]
    ns.h  = function.dotarg('h' ,ns.hbasis)
    ns.h0 = function.dotarg('h0',ns.hbasis)
    ns.δh = '(h - h0) / ?Δt'
    ns.ρ  = ρgeom
    ns.ζ  = ζgeom
    ns.r  = 'ρ ?R0'
    ns.z  = 'h0 ζ'
    ns.x  = function.stack([ns.r,ns.z])

    urbasis = ρdomain.basis('std', degree=urdegree)
    uzbasis = ζdomain.basis('std', degree=uzdegree)


    ns.ubasis  = function.ravel(urbasis[:,_]*uzbasis[_,:], axis=0)
    ns.u       = function.dotarg('u', ns.ubasis)
    ns.upicard = function.dotarg('upicard', ns.ubasis)

    ns.pbasis = ρdomain.basis('std', degree=pdegree)
    ns.p = function.dotarg('p', ns.pbasis)

    usqr    = domain.boundary['inner,top,bottom'].integral('u^2 d:x' @ ns, degree=2*uzdegree)
    ucons   = solver.optimize('u', usqr, droptol=1e-15, arguments={'h0':[h0], 'R0':R0})
    upicard = numpy.zeros_like(ucons)

    ns.π        = numpy.pi
    ns.F        = F
    ns.h0       = h0
    ns.λ        = l
    ns.ν        = nu
    ns.μinf     = muinf
    ns.μ0       = mu0
    ns.δγ       = 'u_,1'
    ns.δγpicard = 'upicard_,1'
    ns.μeff     = 'μinf + (μ0 - μinf) (1 + (λ δγpicard)^2)^((ν - 1) / 2)'
    ns.τ        = 'μeff δγ'
    ns.T        = T
    ns.surf     = surf
    ns.pcap = 'surf / h0'
    ns.Rinit    = Rinit
    ns.hinit    = hinit
    ns.asT      = 'hinit (1 + (8 hinit^2 F T) / (3 π μinf Rinit^4))^(-1 / 4)'


    sample  = ζdomain.sample('gauss',2*uzdegree)*ρdomain.sample('gauss',2*urdegree)
    bsample = ζdomain.sample('gauss',2*uzdegree)*ρdomain.boundary['outer'].sample('gauss',2*urdegree)

    ures  = sample.integral('((ubasis_n r)_,1 τ - (ubasis_n r)_,0 p) d:x' @ ns)
    ures += bsample.integral('(ubasis_n r pcap) d:x' @ ns)

    pres = sample.integral('pbasis_n (-(u r)_,0 - r (δh / h0)) d:x' @ ns)
    hres = domain.boundary['top'].integral('hbasis_n ((F / (?R0^2)) - π p) r d:x' @ ns, degree=2*urdegree)

    Δtsteps = numpy.power(2,range(nt))
    Δtsteps = (T/Δtsteps.sum())*Δtsteps
    Rsteps  = numpy.empty_like(Δtsteps)
    with treelog.iter.plain('timestep',range(nt)) as steps:
        for step in steps:

            Δt = Δtsteps[step]
            times = Δtsteps[:step+1].sum()
            with treelog.iter.plain('picard',range(npicard)) as iterations:
                for iteration in iterations:
                    args = dict(upicard=upicard,h0=[h0],R0=R0,Δt=Δt)
                    state = solver.solve_linear(('u', 'p', 'h'), (ures, pres, hres), constrain=dict(u=ucons), arguments=args)
                    state.update(args)

                    area, μerror= domain.integrate(['r d:x', '(μeff - μinf - (μ0 - μinf) (1 + (λ δγ)^2)^((ν - 1) / 2))^2 r d:x'] @ ns, arguments=state, degree=2*uzdegree)
                    μerror = numpy.sqrt(μerror)/(mu0*area)

                    treelog.user('viscosiy error = {}'.format(μerror))

                    upicard = state['u']

                    ## convergence picard
                    # if step == 0:
                    # original_stdout = sys.stdout
                    # with open("Convergence/viscerr.txt", 'a') as f:
                    #     sys.stdout = f
                    #     print(step+1, iteration+1, μerror)
                    #     sys.stdout = original_stdout

                    if μerror < tol:
                        break
                else:
                    raise RuntimeError('Picard solver did not converge in {} iterations'.format(npicard))  

            postprocess(ρdomain, ζdomain, domain, ns, state)

            Δh = state['h'][0]-state['h0'][0]
            ΔR = -Δh*(R0/h0)

            h0 += Δh
            R0 += ΔR

            Rsteps[step] = R0

            treelog.user('Δh = {}'.format(Δh))
            treelog.user('h  = {}'.format(h0))
            treelog.user('ΔR = {}'.format(ΔR))
            treelog.user('R  = {}'.format(R0))

            with export.mplfigure('radius.png') as fig:
                ax = fig.subplots(2,1)
                ax[0].plot(Δtsteps[:step+1].cumsum(), Rsteps[:step+1])
                ax[0].grid()
                ax[1].loglog(Δtsteps[:step+1].cumsum(), Rsteps[:step+1]-Rinit)
                ax[1].grid()
    ## FEM convergence
    #print('test', ns.asT.eval())
    #err = numpy.sqrt((r_ana-Rsteps[-1])**2)/r_ana
    #treelog.user('err = {}'.format(err))

    # original_stdout = sys.stdout
    # with open('carreauPICARD.txt', 'w') as f:
    #     sys.stdout = f
    #     print('Timestep','Picard','ViscosityError')
    #     sys.stdout = original_stdout
    # print(urdegree, ';', uzdegree, ';', nt, ';', m, ';', tol, ';', err)
    # print('urdegree', ';', 'uzdegree', ';', 'nt', ';', 'm', ';','tol',';', 'err')
    return numpy.concatenate([Δtsteps[:step + 1].cumsum()[:,_], Rsteps[:step + 1,_]],axis=1)

def postprocess(ρdomain, ζdomain, domain, ns, state):

    ns = ns.copy_() # copy namespace so that we don't modify the calling argument

    bezier = domain.sample('bezier', 9)
    x, μeff, u, p = bezier.eval(['x_i', 'μeff', 'u', 'p'] @ ns, **state)

    # # L2 error p bepalen
    # err = domain.integral('(u - asT)^2 d:x' @ ns, degree=uzdegree*2).eval(arguments=state)**.5
    # treelog.user('L2 error: {:.2e}'.format(err))

    with export.mplfigure('viscosity.png') as fig:
      ax = fig.subplots(1,1)
      im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, μeff, shading='gouraud', cmap='jet')
      fig.colorbar(im)

    with export.mplfigure('velocity.png') as fig:
      ax = fig.subplots(1,1)
      im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, u, shading='gouraud', cmap='jet')
      fig.colorbar(im)

    with export.mplfigure('pressure.png') as fig:
      ax = fig.subplots(1,1)
      im = ax.tripcolor(x[:,0], x[:,1], bezier.tri, p, shading='gouraud', cmap='jet')
      fig.colorbar(im)

    # centerline plot
    centerline = ζdomain.sample('uniform',1)*ρdomain.sample('bezier',5) 
    r, u, p = centerline.eval(['r', 'u', 'p'] @ ns, **state)
    # original_stdout = sys.stdout
    # with open('testrallen.txt', 'a') as f:
    #     sys.stdout = f
    #     np.array2string(r, formatter={'float_kind': lambda r: "%.5f" % r})
    #     sys.stdout = original_stdout
    numpy.savetxt("Convergence2/pressurem8.txt", numpy.concatenate([r[:,_], p[:,_]],axis=1), delimiter=',')
    numpy.savetxt("Convergence2/velocityrm8.txt", numpy.concatenate([r[:, _], u[:, _]], axis=1), delimiter=',')
    with export.mplfigure('centerline.png') as fig:
      ax = fig.subplots(2,1)
      ax[0].plot(r, u, label='FE')
      ax[0].set_ylabel('u')
      ax[0].legend()
      ax[1].plot(r, p, label='FE')
      ax[1].set_ylabel('p')
      ax[1].legend()

    # profiles plot
    npoints = 20
    profiles = ρdomain.sample('uniform',1)*ζdomain.sample('bezier',npoints)
    z, u, p = profiles.eval(['z', 'u', 'p'] @ ns, **state)
    numpy.savetxt("Convergence2/velocityzm8.txt", numpy.concatenate([z[:, _], u[:, _]], axis=1), delimiter=',')


    with export.mplfigure('profiles.png') as fig:
      ax = fig.subplots(2,1)
      ax[0].plot(u.reshape(-1,npoints).T, z.reshape(-1,npoints).T)
      ax[1].plot(p.reshape(-1,npoints).T, z.reshape(-1,npoints).T)

#cli.run(main)
cli.choose(main, sample)