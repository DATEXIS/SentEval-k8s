import sys
import json
    
#def print_acc(task):
    #print("{}:".format(task))
    #print("  devacc: {} {} gb".format(a[task]["devacc"], b[task]["devacc"]))
    #print("  acc: {} {} gb".format(a[task]["acc"], b[task]["acc"]))

#def print_pearson_spearman(task):
    #print("{}:".format(task))
    #print("  pearson_mean: {} {} gb".format(a[task]["all"]["pearson"]["mean"], b[task]["all"]["pearson"]["mean"]))
    #print("  spearman_mean: {} {} gb".format(a[task]["all"]["spearman"]["mean"], b[task]["all"]["spearman"]["mean"]))
    #print("  pearson_wmean: {} {} gb".format(a[task]["all"]["pearson"]["wmean"], b[task]["all"]["pearson"]["wmean"]))
    #print("  spearman_wmean: {} {} gb".format(a[task]["all"]["spearman"]["wmean"], b[task]["all"]["spearman"]["wmean"]))

#def print_pearson_spearman_mse(task):
    #print("{}:".format(task))
    #print("  pearson: {} {} gb".format(a[task]["pearson"], b[task]["pearson"]))
    #print("  spearman: {} {} gb".format(a[task]["spearman"], b[task]["spearman"]))
    #print("  devpearson: {} {} gb".format(a[task]["devpearson"], b[task]["devpearson"]))
    #print("  mse: {} {} lb".format(a[task]["mse"], b[task]["mse"]))

def write_result_header(fout):
    fout.write("Model\t")
    fout.write("MR\t")
    fout.write("CR\t")
    fout.write("SUBJ\t")
    fout.write("MPQA\t")
    fout.write("TREC\t")
    fout.write("SST2\t")
    fout.write("SST5\t")
    fout.write("STS12\t")
    fout.write("STS13\t")
    fout.write("STS14\t")
    fout.write("STS15\t")
    fout.write("STS16\t")
    fout.write("SICK-R\t")
    fout.write("STS-B\n")

def write_result_row(fout, name, a):
    fout.write("{}\t".format(name))
    fout.write("{}\t".format(result["MR"]["acc"]))
    fout.write("{}\t".format(result["CR"]["acc"]))
    fout.write("{}\t".format(result["SUBJ"]["acc"]))
    fout.write("{}\t".format(result["MPQA"]["acc"]))
    fout.write("{}\t".format(result["TREC"]["acc"]))
    fout.write("{}\t".format(result["SST2"]["acc"]))
    fout.write("{}\t".format(result["SST5"]["acc"]))
    fout.write("{}\t".format(result["STS12"]["all"]["pearson"]["mean"]))
    fout.write("{}\t".format(result["STS13"]["all"]["pearson"]["mean"]))
    fout.write("{}\t".format(result["STS14"]["all"]["pearson"]["mean"]))
    fout.write("{}\t".format(result["STS15"]["all"]["pearson"]["mean"]))
    fout.write("{}\t".format(result["STS16"]["all"]["pearson"]["mean"]))
    fout.write("{}\t".format(result["SICKRelatedness"]["pearson"]))
    fout.write("{}\n".format(result["STSBenchmark"]["pearson"]))
    
with open(sys.argv[1], "w") as fout:
    write_result_header(fout)
    for filename in sys.argv[2:]:
        with open(filename, "r") as fin:
            result = json.loads(fin.read())
            write_result_row(fout, filename, result)

#with open(sys.argv[1], "r") as f:
    #a = json.loads(f.read())
     
#with open(sys.argv[2], "r") as f:
    #b = json.loads(f.read())
    
#print_acc("MR")
#print_acc("CR")
#print_acc("SUBJ")
#print_acc("MPQA")
#print_acc("TREC")
#print_acc("SST2")
#print_acc("SST5")

#print_pearson_spearman("STS12")
#print_pearson_spearman("STS13")
#print_pearson_spearman("STS14")
#print_pearson_spearman("STS15")
#print_pearson_spearman("STS16")

#print_pearson_spearman_mse("SICKRelatedness")
#print_pearson_spearman_mse("STSBenchmark")
