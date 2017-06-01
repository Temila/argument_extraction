from util import read_file

claims_raw = read_file('New_data/claims.txt', cut=True)
claims = [x[1] for x in claims_raw]
non_claims_raw = read_file('New_data/non_claims.txt')
non_claims = []
for x in non_claims_raw:
    try:
        non_claims.append(x[1])
    except:
        pass
