import pandas as pd
from pathlib import Path
p=Path(__file__).parent
m=pd.read_csv(p/"annexure_industry_mapping_test.csv").fillna("")
o=pd.read_csv(p/"manual_industry_overrides.csv").fillna("")
u=pd.read_csv(p/"universe.csv")
sc=next(x for x in u if x.strip().lower()=="symbol")
qc=next((x for x in u if x.strip().lower()=="series"),None)
if qc: u=u[u[qc].astype(str).str.upper().str.strip()=="EQ"]
valid=set(u[sc].dropna().astype(str).str.upper().str.strip())
known=set(m.Symbol.astype(str).str.upper().str.strip())
need=["Symbol","Sector","Industry","Basic industry","Evidence URL","Evidence summary"]
new=[]
replaced=0
for _,r in o.iterrows():
 s=str(r.Symbol).strip().upper()
 if s not in valid: continue
 if not all(str(r[x]).strip() for x in need) or not str(r["Evidence URL"]).startswith("https://"): raise ValueError("Missing evidence: "+s)
 matches=m.index[m["Symbol"].astype(str).str.upper().str.strip()==s]
 if len(matches):
  i=matches[0]
  if str(m.loc[i,"Sector"]).strip()!="Unclassified": continue
  m.loc[i,["Sector","Industry","Basic industry","Mapping source"]]=[r["Sector"],r["Industry"],r["Basic industry"],"Manual disclosure-verified override; see manual_industry_overrides.csv"]
  replaced+=1
  continue
 new.append(dict(Symbol=s,Sector=r.Sector,Industry=r.Industry,**{"Basic industry":r["Basic industry"]},**{"Mapping source":"Manual disclosure-verified override; see manual_industry_overrides.csv"}))
 known.add(s)
if new or replaced: pd.concat([m,pd.DataFrame(new)],ignore_index=True).to_csv(p/"annexure_industry_mapping_test.csv",index=False)
applied=len(new)+replaced
cp=p/"annexure_unmapped_name_candidates.csv"
if cp.exists():
 candidates=pd.read_csv(cp)
 if "NSE symbol" in candidates: candidates=candidates[~candidates["NSE symbol"].astype(str).str.upper().isin(known)]; candidates.to_csv(cp,index=False)
left=sorted(valid-known)
(p/"annexure_industry_mapping_summary.txt").open("a").write(f"\nDisclosure-backed manual overrides applied: {applied}\nTotal mapped: {len(known)}\nCoverage: {len(known)/len(valid)*100:.1f}%\nUnmapped: {len(left)}\nRemaining symbols: "+", ".join(left)+"\n")
print("Applied",applied,"manual mappings; remaining",len(left))
