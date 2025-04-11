import os
import sys
import numpy as np
import awkward as ak
import pandas as pd
from matplotlib import pyplot as plt
import coffea
from coffea.nanoevents.methods import vector
import mplhep as hep
hep.style.use("CMS")

newdf = sys.argv[1]
newdf = pd.read_csv(newdf)


def getpions(decay_gentau: ak.Array) -> ak.Array :
    ispion_pos = lambda prod: ((prod.pdgId ==  211) | (prod.pdgId ==  321)) # | (prod.pdgId ==  323) | (prod.pdgId ==  325) | (prod.pdgId ==  327) | (prod.pdgId ==  329) )           
    ispion_neg = lambda prod: ((prod.pdgId == -211) | (prod.pdgId == -321)) # | (prod.pdgId == -323) | (prod.pdgId == -325) | (prod.pdgId == -327) | (prod.pdgId == -329) )           

    pions_tau  = decay_gentau[(ispion_pos(decay_gentau) | ispion_neg(decay_gentau))]

    has_three_pions = lambda prod : ak.sum((ispion_pos(prod) | ispion_neg(prod)), axis=1) == 3
    has_two_pos_one_neg_pions = lambda prod : has_three_pions(prod) & (ak.sum(ispion_pos(prod), axis=1) == 2) & (ak.sum(ispion_neg(prod), axis=1) == 1)
    has_two_neg_one_pos_pions = lambda prod : has_three_pions(prod) & (ak.sum(ispion_pos(prod), axis=1) == 1) & (ak.sum(ispion_neg(prod), axis=1) == 2)
    get_sorted_pion_indices   = lambda pions: ak.where(has_two_pos_one_neg_pions(pions),
                                                       ak.argsort(pions.pdgId, ascending=True),
                                                       ak.where(has_two_neg_one_pos_pions(pions),
                                                                ak.argsort(pions.pdgId, ascending=False),
                                                                ak.local_index(pions.pdgId)))

    pions_tau_sorted_indices = get_sorted_pion_indices(pions_tau)
    sorted_pions_tau = pions_tau[pions_tau_sorted_indices]

    return sorted_pions_tau

def getgenpizeros(decay_gentau: ak.Array) -> ak.Array :
    ispizero = lambda col: ((col.pdgId == 111)
                            | (col.pdgId == 311)
                            | (col.pdgId == 130)
                            | (col.pdgId == 310))
    pizeros_tau = decay_gentau[ispizero(decay_gentau)]

    return pizeros_tau

def presel_decay_pis(hcand, hcand_pi):
    #from IPython import embed; embed()
    dummy = hcand_pi[:,:0]
    mask02 = ak.fill_none(ak.firsts((hcand.decayMode >= 0) & (hcand.decayMode <= 2), axis=1), False)
    mask10 = ak.fill_none(ak.firsts(hcand.decayMode >= 10, axis=1), False)
    hcand_pi = ak.where(mask02,
                        hcand_pi[:,0:1],
                        ak.where(mask10,
                                 hcand_pi[:,0:3],
                                 dummy))
    #from IPython import embed; embed()                                                                                                                                               
    return hcand_pi

def presel_decay_pi0s(hcand, hcand_pi0):
    dummy = hcand_pi0[:,:0]
    mask12 = ak.fill_none(ak.firsts(((hcand.decayMode == 1) | (hcand.decayMode == 2)), axis=1), False)
    hcand_pi0 = ak.where(mask12,
                         hcand_pi0[:,0:1],
                         dummy)
    return hcand_pi0

def reArrangeDecayProducts(
        events: ak.Array,
        **kwargs
) :
    hcand      = events.hcand
    hcandprod  = events.hcandprod

    hcand1     = hcand[:, 0:1]
    hcand2     = hcand[:, 1:2]
    hcand1prod = hcandprod[:,0]
    hcand2prod = hcandprod[:,1]

    hcand1prod_pions = getpions(hcand1prod)
    hcand2prod_pions = getpions(hcand2prod)

    hcand1prod_pizeros = getgenpizeros(hcand1prod)
    hcand2prod_pizeros = getgenpizeros(hcand2prod)

    # hcand1 and its decay products                                                                                                                                                   
    p4_hcand1     = ak.with_name(hcand1, "PtEtaPhiMLorentzVector")
    p4_hcand1_pi  = ak.with_name(hcand1prod_pions, "PtEtaPhiMLorentzVector")
    p4_hcand1_pi0 = ak.with_name(hcand1prod_pizeros, "PtEtaPhiMLorentzVector")

    # hcand2 and its decay products                                                                                                                                                   
    p4_hcand2     = ak.with_name(hcand2, "PtEtaPhiMLorentzVector")
    p4_hcand2_pi  = ak.with_name(hcand2prod_pions, "PtEtaPhiMLorentzVector")
    p4_hcand2_pi0 = ak.with_name(hcand2prod_pizeros, "PtEtaPhiMLorentzVector")

    hcand1AndProds = ak.concatenate([p4_hcand1, p4_hcand1_pi, p4_hcand1_pi0], axis=1)
    hcand2AndProds = ak.concatenate([p4_hcand2, p4_hcand2_pi, p4_hcand2_pi0], axis=1)

    return {"p4h1"       : p4_hcand1,
            "p4h1pi"     : p4_hcand1_pi,
            "p4h1pi0"    : p4_hcand1_pi0,
            "p4h2"       : p4_hcand2,
            "p4h2pi"     : p4_hcand2_pi,
            "p4h2pi0"    : p4_hcand2_pi0}

def reArrangeGenDecayProducts(
        events: ak.Array,
        **kwargs
) :
    ghcand       = events.GenTau
    ghcandprod   = events.GenTauProd

    hcand1     = ghcand[:, 0:1]
    hcand2     = ghcand[:, 1:2]
    hcand1prod = ghcandprod[:,0]
    hcand2prod = ghcandprod[:,1]


    hcand1prod_pions = getpions(hcand1prod)
    hcand2prod_pions = getpions(hcand2prod)

    hcand1prod_pizeros = getgenpizeros(hcand1prod)
    hcand2prod_pizeros = getgenpizeros(hcand2prod)

    # hcand1 and its decay products                                                                                                                                                   
    p4_hcand1     = ak.with_name(hcand1, "PtEtaPhiMLorentzVector")
    p4_hcand1_pi  = ak.with_name(hcand1prod_pions, "PtEtaPhiMLorentzVector")
    p4_hcand1_pi  = presel_decay_pis(p4_hcand1, p4_hcand1_pi) # safe                                                                                                                  
    p4_hcand1_pi0 = ak.with_name(hcand1prod_pizeros, "PtEtaPhiMLorentzVector")
    p4_hcand1_pi0 = presel_decay_pi0s(p4_hcand1, p4_hcand1_pi0) # safe                                                                                                                

    # hcand2 and its decay products                                                                                                                                                   
    p4_hcand2     = ak.with_name(hcand2, "PtEtaPhiMLorentzVector")
    p4_hcand2_pi  = ak.with_name(hcand2prod_pions, "PtEtaPhiMLorentzVector")
    p4_hcand2_pi  = presel_decay_pis(p4_hcand2, p4_hcand2_pi)   # safe                                                                                                                
    p4_hcand2_pi0 = ak.with_name(hcand2prod_pizeros, "PtEtaPhiMLorentzVector")
    p4_hcand2_pi0 = presel_decay_pi0s(p4_hcand2, p4_hcand2_pi0) # safe                                                                                                                

    hcand1AndProds = ak.concatenate([p4_hcand1, p4_hcand1_pi, p4_hcand1_pi0], axis=1)
    hcand2AndProds = ak.concatenate([p4_hcand2, p4_hcand2_pi, p4_hcand2_pi0], axis=1)

    #from IPython import embed; embed()                                                                                                                                               

    return {"p4h1"        : p4_hcand1,
            "p4h1pi"      : p4_hcand1_pi,
            "p4h1pi0"     : p4_hcand1_pi0,
            "p4h2"        : p4_hcand2,
            "p4h2pi"      : p4_hcand2_pi,
            "p4h2pi0"     : p4_hcand2_pi0}


def plotit(phicp_list, wt_list, tag, bins=None, labels=[]):
    opath = "output"
    if os.path.exists(opath):
        print(f"{opath} dir found")
    else:
        os.mkdir(opath)

    if bins == None:
        bins = np.linspace(0.0, 2*np.pi, 50)
    else:
        bins = np.linspace(bins[0], bins[1], bins[2])

    # plot
    plt.figure(figsize=(6.9, 4.8))
    # Add CMS-style text
    hep.cms.label(rlabel="")
    #hep.cms.text("Simulation", loc=1)

    if len(wt_list) == 3:
        plt.hist(phicp_list[0], bins=bins, histtype='step', weights=wt_list[0], linewidth=2, label='CP even')
        plt.hist(phicp_list[0], bins=bins, histtype='step', weights=wt_list[1], linewidth=2, label='CP odd')
        plt.hist(phicp_list[0], bins=bins, histtype='step', weights=wt_list[2], linewidth=2, label='CP mix')
    else:
        for i in range(len(phicp_list)):
            lab = labels[i]
            plt.hist(phicp_list[i], bins=bins, histtype='step', linewidth=2, label=lab)
            
    plt.legend()
    #plt.ylim(0.0,1000.0)
    #plt.title(f"{tag}")
    plt.xlabel(f"{tag}")
    plt.tight_layout()
    plt.savefig(f"{opath}/{tag}.png", dpi=300)


from PhiCP_Estimator import GetPhiCP

file = "/pbs/home/o/oponcet/private/TauRegression/BasicNN/model/merged_rhorho_Apr06_v2.parquet"
events = ak.from_parquet(file)

# pure Gen
print("All is Gen")
P4_gen_dict = reArrangeGenDecayProducts(events)
P4_gen_dict = {key:ak.Array(val, behavior=coffea.nanoevents.methods.vector.behavior) for key,val in P4_gen_dict.items()}
phicp_gen_dict = GetPhiCP(P4_gen_dict, "PV", "PV",  "rho",  "rho" )

# pure Reco
print("All is Reco")
P4_reco_dict = reArrangeDecayProducts(events)
P4_reco_dict = {key:ak.Array(val, behavior=coffea.nanoevents.methods.vector.behavior) for key,val in P4_reco_dict.items()}
phicp_reco_dict = GetPhiCP(P4_reco_dict, "PV", "PV",  "rho",  "rho" )
phicp_reco_dict_DP = GetPhiCP(P4_reco_dict, "DP", "DP",  "rho",  "rho" )

# GenTau - rest Reco
print("GenTau - rest of all is Reco")
gentau1 = P4_gen_dict["p4h1"]
gentau2 = P4_gen_dict["p4h2"]
P4_gen_reco_dict = P4_reco_dict.copy()
P4_gen_reco_dict["p4h1"] = gentau1
P4_gen_reco_dict["p4h2"] = gentau2
phicp_gen_reco_dict = GetPhiCP(P4_gen_reco_dict, "PV", "PV",  "rho",  "rho" )


print("Extracting spinner weights ...")
cpeven_wt = ak.to_numpy(events.tauspinner_weight_cpeven)
cpodd_wt = ak.to_numpy(events.tauspinner_weight_cpodd)
cpmm_wt = ak.to_numpy(events.tauspinner_weight_cpmixm)
wt_list = [cpeven_wt, cpodd_wt, cpmm_wt]


print("Plotting PhiCPs ...")
plotit([phicp_gen_dict['phicp']], wt_list, "PhiCP_Gen")
plotit([phicp_reco_dict['phicp']], wt_list, "PhiCP_Reco")
plotit([phicp_gen_reco_dict['phicp']], wt_list, "PhiCP_Gen_Reco")

#pt1  = ak.from_numpy(newdf["pt_tau_1_regress"].to_numpy())[:,None]
pt1  = ak.from_numpy(newdf["pt1"].to_numpy())[:,None]
eta1  = ak.from_numpy(newdf["eta1"].to_numpy())[:,None]
phi1  = ak.from_numpy(newdf["phi1"].to_numpy())[:,None]
#eta1 = ak.Array(newdf["eta1"])[:,None]
#phi1 = ak.Array(newdf["phi1"])[:,None]

#pt2  = ak.from_numpy(newdf["pt_tau_2_regress"].to_numpy())[:,None]
pt2  = ak.from_numpy(newdf["pt2"].to_numpy())[:,None]
eta2  = ak.from_numpy(newdf["eta2"].to_numpy())[:,None]
phi2  = ak.from_numpy(newdf["phi2"].to_numpy())[:,None]
#eta2 = ak.Array(newdf["eta2"])[:,None]
#phi2 = ak.Array(newdf["phi2"])[:,None]




P4_reg_dict = P4_reco_dict.copy()

p4h1 = P4_reg_dict["p4h1"]
p4h1 = ak.with_field(p4h1, pt1, "pt")
# p4h1 = ak.with_field(p4h1, eta1, "eta")
# p4h1 = ak.with_field(p4h1, phi1, "phi")
# p4h1 = ak.with_field(p4h1, ak.ones_like(phi1)*1.777, "mass") # ADD THIS


p4h2 = P4_reg_dict["p4h2"]
p4h2 = ak.with_field(p4h2, pt2, "pt")
# p4h2 = ak.with_field(p4h2, eta2, "eta")
# p4h2 = ak.with_field(p4h2, phi2, "phi")
# p4h2 = ak.with_field(p4h2, ak.ones_like(phi2)*1.777, "mass") # ADD THIS

P4_reg_dict["p4h1"] = p4h1
P4_reg_dict["p4h2"] = p4h2

phicp_reg_dict = GetPhiCP(P4_reg_dict, "PV", "PV",  "rho",  "rho" )

print("Plotting Regressed PhiCPs ...")
plotit([phicp_reg_dict['phicp']], wt_list, "PhiCP_Reg")


plotit([P4_reg_dict["p4h1"].pt,
        P4_gen_dict["p4h1"].pt,
        P4_reco_dict["p4h1"].pt], [], "Leading_tau_Pt", bins=[30.0,230.0,100], labels=['Reg','Gen','Reco'])
plotit([P4_reg_dict["p4h2"].pt,
        P4_gen_dict["p4h2"].pt,
        P4_reco_dict["p4h2"].pt], [], "Sub_leading_tau_Pt", bins=[30.0,230.0,100], labels=['Reg','Gen','Reco'])


gen_mass = (P4_gen_dict["p4h1"] + P4_gen_dict["p4h2"]).mass
reco_mass = (P4_reco_dict["p4h1"] + P4_reco_dict["p4h2"]).mass
reg_mass = (P4_reg_dict["p4h1"] + P4_reg_dict["p4h2"]).mass

plotit([gen_mass,
        reco_mass,
        reg_mass], [], "Invariant_mass", bins=[0.0,300.0,150], labels=['Gen','Reco','Reg'])


plotit([(P4_gen_dict["p4h1"].pt - P4_reco_dict["p4h1"].pt)/P4_gen_dict["p4h1"].pt,
        (P4_gen_dict["p4h1"].pt - P4_reg_dict["p4h1"].pt)/P4_gen_dict["p4h1"].pt], [], "Pt_resolution", bins=[-2.0,2.0,100], labels=['Gen-Reco','Gen-Reg'])

