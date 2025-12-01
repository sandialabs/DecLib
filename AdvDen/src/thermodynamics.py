from math import exp, log, pow

#        self.Cv = 717.0
#        self.Cp = 1004.0
#        self.R = 287.0
#        self.pr = 1000.0 * 100.
#        self.Tr = 273.15


class EmptyThermo():
    def __init__(self, ic):

        self.Cv = None
        self.Cp = None
        self.R = None
        self.pr = None
        self.Tr = None
        self.gamma = None
        self.kappa = None
        self.delta = None

class IdealGasEntropy():
    def __init__(self, ic):

        self.Cv = ic.Cv
        self.Cp = ic.Cp
        self.R = ic.R
        self.pr = ic.pr
        self.Tr = ic.Tr
        self.gamma = self.Cp / self.Cv
        self.kappa = self.R / self.Cp
        self.delta = self.R / self. Cv
        
    def compute_dudalpha(self, alpha, eta):
        u = self.compute_u(alpha, eta)
        return - self.R / self.Cv * u  / alpha
        
#    real U = compute_U(alpha, entropic_var, qd, qv, ql, qi);
#    return -cst.Rd / cst.Cvd * U / alpha;
    def compute_dudeta(self, alpha, eta):
        u = self.compute_u(alpha, eta)
        return u / self.Cv
        
#    real U = compute_U(alpha, entropic_var, qd, qv, ql, qi);
#    return U / cst.Cvd;
    def compute_u(self, alpha, eta):
        return self.Cv * self.Tr * pow(alpha * self.pr / (self.R * self.Tr), -self.delta) * exp(eta / self.Cv)

	#OTHER STUFF NEEDED MAYBE?
#cst.Cvd * cst.Tr *
#   pow(alpha * cst.pr / (cst.Rd * cst.Tr), -cst.delta_d) *
#   exp(entropic_var / cst.Cvd);

class IdealGasEntropyAbgrall():
    def __init__(self, ic):
        self.gamma = ic.gamma
        self.Cv = ic.Cv
    
    def compute_dudrho(self, rho, eta):
        return pow(rho, self.gamma - 2.0) * exp(eta / self.Cv)
        
    def compute_dudeta(self, rho, eta):
        return pow(rho, self.gamma - 1.0) * exp(eta / self.Cv) / (self.gamma - 1.0) / self.Cv
        
    def compute_u(self, rho, eta):
        return pow(rho, self.gamma - 1.0) * exp(eta / self.Cv) / (self.gamma - 1.0)
        
    def get_eta(self, rho, p):
        return self.Cv * log(p / pow(rho, self.gamma))
    
    def get_p(self, rho, eta):
        return pow(rho, self.gamma) * exp(eta / self.Cv)
        
    def get_T(self, rho, eta):
        return pow(rho, self.gamma - 1.0) * exp(eta / self.Cv) / (self.gamma - 1.0) / self.Cv
      
def getThermo(params, ic):
    if params['thermo'] == 'idealgasentropyabgrall':
        thermodynamics = IdealGasEntropyAbgrall(ic)
    if params['thermo'] == 'EmptyThermo':
        thermodynamics = EmptyThermo(ic)
    return thermodynamics
