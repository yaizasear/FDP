import os
import tarfile
import numpy as np
import subprocess
from datasets import get_protein_sequences


def makeblastdb(config, jobid):
    # Write fasta
    db = []
    db_path = os.path.join(config["out_dir"], "proteinGAN_" + jobid, "BLAST_DB")
    db_fasta = db_path + ".fa"
    files = [config["train_data"], config["val_data"]]
    for file_path in files:
        with tarfile.open(file_path, 'r:gz') as tar:
            member = tar.getmembers()[-1]
            f = tar.extractfile(member)
            lines = f.readlines()[1:]
            for i, line in enumerate(lines):
                seq = line.split()[-1].decode("utf-8")
                record = "{}{}{}{}".format(">", file_path.split("/")[-1] + "." + str(i), os.linesep, seq)
                db.append(record)
    with open(db_fasta, "w+") as f:
        f.write(os.linesep.join(db))
    
    # Run makeblastdb
    db_process = subprocess.Popen(
        ["makeblastdb", "-in", db_fasta, "-input_type", "fasta", "-dbtype", "prot", 
        "-out", db_path, "-parse_seqids"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    db_process.wait()

def runblast(fake_samples, step, jobid, config):
    # Write fasta file
    query_path = os.path.join(config["out_dir"], "proteinGAN_" + jobid, "seqs_" + str(step) + ".fasta")
    fasta_sequences = get_protein_sequences(fake_samples)
    with open(query_path, "w+") as f:
        f.write(os.linesep.join(fasta_sequences))

    # Run blastp
    db_path = os.path.join(config["out_dir"], "proteinGAN_" + jobid, "BLAST_DB")
    blastp_process = subprocess.Popen(
        ["blastp", "-db", db_path, "-outfmt", "10 qseqid pident", "-max_target_seqs", "1",
         "-matrix", "BLOSUM45", "-query", query_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Parse results
    results, _ = blastp_process.communicate()
    results = results.decode()
    parsed_results = {} # a dictonary where key is qseqid, value is pident
    for line in results.split(os.linesep):
        if line != "":
            parts = line.split(",")
            parsed_results[parts[0]] = parts[1]
    
    # Calculate score
    if parsed_results != {}:
        identities = np.zeros(config["batch_size"])
        for i, k in enumerate(parsed_results):
            identities[i] = float(parsed_results[k])
        q75 = int(config["batch_size"] * 0.75)
        avg_identity = np.mean(sorted(identities)[q75:])
        max_identity = np.max(identities)
        if config["min_identity"] <= avg_identity <= config["max_identity"]:
            return -config["blast_percentage"], avg_identity, max_identity 
        else:
            return config["blast_percentage"], avg_identity, max_identity

    return config["blast_percentage"], 0, 0
