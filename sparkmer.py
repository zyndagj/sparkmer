#!/usr/bin/env python
from pyspark import SparkContext, SparkConf, StorageLevel
import sys, argparse
from pyspark.mllib.feature import IDF, HashingTF
from pyspark.mllib.linalg import SparseVector
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.spatial.distance import squareform
from string import maketrans
import sparseDist
import time
#from pyspark.ml.feature import NGram
#from pyspark.sql import SQLContext, DataFrame, Row

import scipy
from scipy import sparse
def _my_get_index_dtype(*a, **kw):
	#kw.pop('check_contents', None)
	return np.int64
sparse.sputils.get_index_dtype = _my_get_index_dtype
sparse.compressed.get_index_dtype = _my_get_index_dtype
sparse.csr.get_index_dtype = _my_get_index_dtype
sparse.csr_matrix.get_index_dtype = _my_get_index_dtype
sparse.bsr.get_index_dtype = _my_get_index_dtype

def main():
	parser = argparse.ArgumentParser(description="sparK-mer")
	parser.add_argument("-N",metavar="INT", help="Number of nodes to use [%(default])", default=19, type=int)
	parser.add_argument("-C",metavar="INT", help="Cores per node [%(default)]", default=24, type=int)
	parser.add_argument("-E",metavar="INT", help="Cores per executor [%(default)]", default=4, type=int)
	parser.add_argument("-M",metavar="STR", help="Namenode", default="c252-104", type=str)
	parser.add_argument("-L",metavar="STR", help="Log level", default="WARN", type=str)
	parser.add_argument("-K",metavar="INT", help="k-mer size [%(default)]", default=15, type=int)
	parser.add_argument("-v", action="store_true", help="Verbose output")
	args = parser.parse_args()
	
	executorInstances = args.N*args.C/args.E

	# Explicitly set the storage level
	#StorageLevel(True, True, False, True, 1)
	
	# Set up spark configuration
	conf = SparkConf().setMaster("yarn-client").setAppName("sparK-mer")
	#conf = SparkConf().setMaster("local[16]").setAppName("sparK-mer")
	conf.set("yarn.nodemanager.resource.cpu_vcores",args.C)
	# Saturate with executors
	conf.set("spark.executor.instances",executorInstances)
	conf.set("spark.executor.heartbeatInterval","5s")
	# cores per executor
	conf.set("spark.executor.cores",args.E)
	# set driver cores
	conf.set("spark.driver.cores",12)
	# Number of akka threads
	conf.set("spark.akka.threads",256)
	# Agregation worker memory
	conf.set("spark.python.worker.memory","5g")
	# Maximum message size in MB
	conf.set("spark.akka.frameSize","128")
	conf.set("spark.akka.timeout","200s")
	conf.set("spark.akka.heartbeat.interval","10s")
	#conf.set("spark.broadcast.blockSize","128m")
	conf.set("spark.driver.maxResultSize", "20g")
	conf.set("spark.reducer.maxSizeInFlight","5g")
	conf.set("spark.executor.memory","7g")
	#conf.set("spark.shuffle.memoryFraction",0.4)
	#conf.set("spark.storage.memoryFraction",0.3)
	#conf.set("spark.storage.unrollFraction",0.3)
	#conf.set("spark.storage.memoryMapThreshold","256m")
	#conf.set("spark.kryoserializer.buffer.max","1g")
	#conf.set("spark.kryoserializer.buffer","128m")
	#conf.set("spark.core.connection.ack.wait.timeout","600")
	#conf.set("spark.shuffle.consolidateFiles","true")
	#conf.set("spark.shuffle.file.buffer","32m")
	conf.set("spark.shuffle.manager","sort")
	conf.set("spark.shuffle.spill","true")

	# Set up Spark Context
	sc = SparkContext("","",conf=conf)
	sc.setLogLevel(args.L)

	# Process DB
	#frequencyProfile = generateFP(sc, args.K, "hdfs://c252-104/user/gzynda/random_20", args.v)
	fpStart = time.time()
	frequencyProfile = generateFP(sc, args.K, "/user/gzynda/library", args.v)
	frequencyProfile.cache()
	nGenomes = frequencyProfile.count()
	fpSecs = time.time()-fpStart
	print "############################################"
	print "Counted %i genomes in %.2f seconds"%(nGenomes, fpSecs)
	print "############################################"

	# Parse FQ
	fqStart = time.time()
	fqFrequency = parseFQ(sc, args.K, "/user/gzynda/reads/HiSeq_accuracy.fq", args.v)
	fqFrequency.cache()
	nReads = fqFrequency.count()
	fqSecs = time.time()-fqStart
	print "############################################"
	print "Parsed %i reads in %.2f seconds"%(nReads, fqSecs)
	print "############################################"

	# Classify reads
	classStart = time.time()
	#classify(sc, fqFrequency, frequencyProfile, args.v)
	nReads = setClassify(sc, fqFrequency, frequencyProfile, args.v)
	classSecs = time.time()-classStart
	print "############################################"
	print "Classified %i reads in %.2f seconds"%(nReads, classSecs)
	print "Ran on %i executor instances"%(executorInstances)
	print "K = %i"%(args.K)
	print "############################################"
	sys.exit()

bases = 'AGCT'
baseDict = dict(zip(bases, map(str,range(4))))
def kmer2index(kmer):
	'''
	Transforms a kmer into its index. While hashing may
	be more efficient it removes the possibility of collisions.

	>>> kmer2index('AAA')
	0
	'''
	return int(''.join(map(lambda x: baseDict[x], kmer)),4)

def makeDendro(pDist, metric, method, prefix):
	'''
	Generate and plot a dendrogram from a distance matrix.
	'''
	Z = linkage(squareform(pDist), method=method)
	plt.figure(figsize=(8,13))
	dendrogram(Z, labels=labels, orientation="left")
	plt.tight_layout(rect=[0.0, 0.0, 1.0, 1.0])
	plt.savefig('%s_%s_%s.png'%(prefix,metric,method))
	plt.close()

def plotParts(rdd, pName, title="", useLog=False):
	recsPerPart = rdd.mapPartitions(lambda x: (yield sum(1 for i in x)))
	plt.figure()
	n, bins, patches = plt.hist(recsPerPart.collect(), bins=20, log=useLog)
	if useLog: plt.ylim(ymin=1)
	if useLog:
		plt.ylabel("log(# Partitions)")
	else:
		plt.ylabel("# Partitions")
	plt.xlabel("Records on Partition")
	if title: plt.title(title)
	plt.savefig(pName+".png")
	plt.close()

def filterFA(line):
	'''
	Filters the header and comments from
	a faster file.
	'''
	return (line and line[0] not in ">#")

def chunk(seq, k=3, c=1000):
	'''
	Chunks and a fasta sequence and its reverse complement
	into smaller, overlapping sequences for kmer
	calculations.
	'''
	seqLen = len(seq)
	fChunks = [seq[i:min(i+c+k, seqLen)] for i in xrange(0, seqLen, c)]
	revSeq = revcomp(seq)
	rChunks = [revSeq[i:min(i+c+k, seqLen)] for i in xrange(0, seqLen, c)]
	return fChunks+rChunks

def index2csr(indices, kmer):
	'''
	Creates a csr index from a list of indices.
	'''
	sparse.sputils.get_index_dtype = _my_get_index_dtype
	sparse.compressed.get_index_dtype = _my_get_index_dtype
	sparse.csr.get_index_dtype = _my_get_index_dtype
	sparse.csr_matrix.get_index_dtype = _my_get_index_dtype
	sparse.bsr.get_index_dtype = _my_get_index_dtype
	nVals = 4**kmer
	numInd = len(indices)
	data = np.ones(numInd)
	inds = np.array(indices, dtype=np.int64)
	indptr = np.array([0,numInd], dtype=np.int64)
	V = sparse.csr_matrix((data, inds, indptr), shape=(1,nVals), dtype=np.float64)
	V.sum_duplicates()
	return V

revTab = maketrans('AGCT','TCGA')
def revcomp(kmerSeq):
	return kmerSeq[::-1].translate(revTab)

def indexListFromSeq(seq, kmer):
	kmerList = [seq[i:i+kmer] for i in xrange(len(seq)-kmer)]
	filteredList = filter(lambda x: len(x.translate(None,'AGCT')) == 0, kmerList)
	return map(kmer2index, filteredList)

def generateFP(sc, kmer, dbLocation, v):
	# Read Fasta Files
	fastas = sc.wholeTextFiles(dbLocation, minPartitions=3000, use_unicode=False)#.repartition(2000)
	# Split lines, filter header sequences, and join
	#fileFaSeq = fastas.mapValues(lambda x: ''.join(filter(filterFA, x.split('\n'))))
	#fileFaSeq = fastas.mapValues(lambda x: index2csr(indexListFromSeq(''.join(filter(filterFA, x.split('\n'))),kmer),kmer))
	fileFaSeq = fastas.mapValues(lambda x: set(indexListFromSeq(''.join(filter(filterFA, x.split('\n'))),kmer)))
	return fileFaSeq
	if v:
		print "Read %i files from %s"%(fileFaSeq.count(), dbLocation)
		for fName, seq in fileFaSeq.collect():
			print fName, '%ibp'%(len(seq)), seq[:50]
	# Chunk into "c" bp chunks with kmer overlap
	chunked = fileFaSeq.flatMapValues(lambda x: chunk(x, kmer, c=1000))\
		.repartition(fileFaSeq.getNumPartitions()*2)
	if v:
		print "Split sequences and distributed into %i partitions"%(chunked.getNumPartitions())
		for fName, seq in chunked.take(2):
			print fName, '%ibp'%(len(seq)), seq[:50]
	csrChunks = chunked.mapValues(lambda x: index2csr(indexListFromSeq(x, kmer), kmer))
	if v:
		for fName, csr in csrChunks.take(5):
			print fName, csr.size, csr.sum(), csr.shape
	# Reduce counts from same file together
	totalCSR = csrChunks.reduceByKey(lambda x,y: x+y)
	if v:
		for fName, csr in totalCSR.take(5):
			print fName, csr.size, csr.sum(), csr.shape
	return totalCSR

def parseFQ(sc, kmer, fqLocation, v):
	i0 = sc.textFile(fqLocation, use_unicode=False).zipWithIndex().filter(lambda (x,y): y%4 < 2)
	i0.cache()
	headers = i0.filter(lambda (x,y): y%4 == 0).map(lambda (x,y): (y,x))
	seqs = i0.filter(lambda (x,y): y%4 == 1).map(lambda (x,y): (y-1,x))
	#csrReads = headers.join(seqs).map(lambda (i,(n,s)): (i/4, (n, index2csr(indexListFromSeq(s, kmer), kmer))))
	csrReads = headers.join(seqs).map(lambda (i,(n,s)): (i/4, (n, set(indexListFromSeq(s, kmer)))))
	i0.unpersist()
	if v:
		for i, v in csrReads.take(5):
			fName, csr = v
			print fName, csr.size, csr.sum(), csr.shape
	nParts = csrReads.getNumPartitions()
	if nParts < 120: nParts = 120
	#plotParts(csrReads, "readsDefault", "Default Partitioning of Reads")
	csrReads = csrReads.partitionBy(nParts, partitionFunc=lambda x: x)
	#plotParts(csrReads, "readsManual_2", "Manual Partitioning of Reads")
	return csrReads

def minDist(a,b):
	if a[1] < b[1]:
		return a
	return b

def maxDist(a,b):
	if a[2] > b[2]:
		return a
	else:
		return b

def printTake(rdd):
	for r in rdd.take(5):
		print r

def toSet(csr):
	return set(csr.indices)

def setJS(sA, sB):
	iAB = len(sA.intersection(sB))
	uAB = len(sA.union(sB))
	return float(iAB)/float(uAB)

def setClassify(sc, setReads, FP, v):
	# csrReads = [(index, (name, csr)), (index, (name, csr)), ...]
	# FP = [(name, csr), (name, csr), ...]
	readFP = setReads.map(lambda x: x[1][1]).reduce(lambda x,y: x|y)
	rfp = sc.broadcast(readFP)
	jD = FP.map(lambda (g,gc): (g, gc, setJS(gc, rfp.value)))
	sortI = jD.sortBy(lambda x: x[2], ascending=False).zipWithIndex() # largest first
	topN = 150
	top100 = sortI.filter(lambda (x,y): y < topN) # top 100
	top100G = top100.map(lambda (x,y): (y,(x[0], x[1]))).partitionBy(topN/2, lambda x: x)
	#plotParts(top100G, "topManual_2", "Manual Partitioning of Pairwise Data")
	cart = setReads.cartesian(top100G).map(lambda ((ri, rv),(gi, gv)): (ri, rv+gv))
	#plotParts(cart, "cartesianManual_2", "Manual Partitioning of Pairwise Data")
	jaccard = cart.map(lambda (i,(rn,rv,gn,gv)): (i, (rn, gn, setJS(rv, gv))))
	#plotParts(jaccard, "jaccardManual_2", "Manual Partitioning of Jaccard Distances")
	maxJ = jaccard.reduceByKey(maxDist)
	##write out
	#maxJ.map(lambda (i,v): v[0]+'\t'+v[1].split('/')[-1]).saveAsTextFile("/user/gzynda/class_reads")
	return maxJ.count()

def classify(sc, csrReads, FP, v):
	# csrReads = [(index, (name, csr)), (index, (name, csr)), ...]
	# FP = [(name, csr), (name, csr), ...]
	readFP = csrReads.map(lambda x: x[1][1]).reduce(lambda x,y: x+y)
	rfp = sc.broadcast(readFP)
	jD = FP.map(lambda (g,gc): (g, gc, sparseDist.jaccard(gc, rfp.value)))
	sortI = jD.sortBy(lambda x: x[2], ascending=False).zipWithIndex() # largest first
	top100 = sortI.filter(lambda (x,y): y<200) # top 100
	top100G = top100.map(lambda (x,y): (y,(x[0], x[1]))).partitionBy(100, lambda x: x)
	#plotParts(top100G, "topManual_2", "Manual Partitioning of Pairwise Data")
	cart = csrReads.cartesian(top100G).map(lambda ((ri, rv),(gi, gv)): (ri, rv+gv))
	#plotParts(cart, "cartesianManual_2", "Manual Partitioning of Pairwise Data")
	jaccard = cart.map(lambda (i,(rn,rv,gn,gv)): (i, (rn, gn, sparseDist.jaccard(rv, gv))))
	plotParts(jaccard, "jaccardManual_2", "Manual Partitioning of Jaccard Distances")
	maxJ = jaccard.reduceByKey(maxDist)
	##write out
	maxJ.map(lambda (i,v): v[0]+'\t'+v[1].split('/')[-1]).saveAsTextFile("/user/gzynda/class_reads.tab")
	sys.exit()
	for i in maxJ.map(lambda (i,v): v).collect():
		read, ref, d = i
		print "%20s   %20s   %.5f"%(read, ref.split('/')[-1], d)
	#maxJ = jaccard.groupByKey(numPartitions=1000)
	#maxJ = jaccard.repartition(1000).sortByKey().reduceByKeyLocally(lambda x,y: x+y)
	#maxJ = jaccard.foldByKey(0, lambda x,y: x+y)
	#printTake(maxJ)
	sys.exit()

if __name__ == "__main__":
	main()
