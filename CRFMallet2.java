/* Copyright (C) 2003 University of Pennsylvania.
   This file is part of "MALLET" (MAchine Learning for LanguagE Toolkit).
http://www.cs.umass.edu/~mccallum/mallet
This software is provided under the terms of the Common Public License,
version 1.0, as published by http://www.opensource.org.  For further
information, see the file `LICENSE' included with this distribution. */

package edu.mit.csail.spatial.learner;

import cc.mallet.fst.*;
import java.util.HashMap;
import java.io.File;
import java.io.FileReader;
import java.io.FileInputStream;
import java.io.ObjectInputStream;
import java.io.Reader;
import java.io.FileNotFoundException;
import java.io.IOException;
import cc.mallet.types.SparseVector;
import java.util.regex.Pattern;
import java.util.List;
import java.util.ArrayList;

import cc.mallet.types.Alphabet;
import cc.mallet.types.AugmentableFeatureVector;
import cc.mallet.types.FeatureVector;
import cc.mallet.types.FeatureVectorSequence;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import cc.mallet.types.LabelAlphabet;
import cc.mallet.types.LabelSequence;
import cc.mallet.types.Sequence;
import cc.mallet.types.TokenSequence;
import cc.mallet.types.Token;
import cc.mallet.fst.semi_supervised.CRFTrainerByEntropyRegularization;

import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.Pipe;
import cc.mallet.pipe.iterator.LineGroupIterator;
import cc.mallet.pipe.SvmLight2FeatureVectorAndLabel;


/**
 * This class's main method trains, tests, or runs a generic CRF-based
 * sequence tagger.
 * <p>
 * Training and test files consist of blocks of lines, one block for each instance, 
 * separated by blank lines. Each block of lines should have the first form 
 * specified for the input of {@link CRFMalletSentence2FeatureVectorSequence}. 
 * A variety of command line options control the operation of the main program, as
 * described in the comments for {@link #main main}.
 *
 * @author Fernando Pereira <a href="mailto:pereira@cis.upenn.edu">pereira@cis.upenn.edu</a>
 * @version 1.0
 */
public class CRFMallet2
{
    /**
     * No <code>CRFMallet</code> objects allowed.
     */
    
    //parameters to the CRF learning
//    protected  String defaultLabel = "False";
    protected  String defaultLabel = "label_start";
    protected  int iterations = 500;
    //variance parameter
    protected  double sigma = 10.0;
    protected  int cacheSize = 100000;

    private String classifier = "";
    private List<String> classifiers;

    //the CRF and the dataset
    protected  CRF crf = null;
    protected  InstanceList mTrainingData = null;
    
    public CRF getCrf() {
        return crf;
    }

    public void setClassifier(String name) {
        if (! classifiers.contains(name)) {
            throw new IllegalArgumentException("Bad classifier type: " + name);
        }
        classifier = name;
    }
    public String getClassifier() {
        return classifier;
    }

    public void setTrainingIterations(int i) {
	iterations = i;
    }
    public int getTrainingIterations() {
	return iterations;
    }


    private void initialize() {
        classifiers = new ArrayList<String>();
        classifiers.add("CRFTrainerByLabelLikelihood");
        classifiers.add("CRFTrainerByL1LabelLikelihood");
        classifiers.add("CRFTrainerByEntropyRegularization");
    }

    public CRFMallet2() {
        initialize();
    }

        

    //initialization
    public CRFMallet2(String datafilename, double sigma) throws FileNotFoundException {
        try {
            initialize();
            //get the training data
            this.mTrainingData = getTrainingData(datafilename);
            this.sigma = sigma;
            
            //get the crf
            this.crf = new CRF(mTrainingData.getPipe(), (Pipe) null);
        } catch (RuntimeException e) {
            e.printStackTrace();
            throw e;
        }
    }


    public InstanceList getTrainingData(String filename)
	throws FileNotFoundException{
        System.out.println("Loading " + filename);
	Reader trainingFile = new FileReader(new File(filename));

        ArrayList<Pipe> pipeList = new ArrayList<Pipe>();
        pipeList.add(new SvmLight2FeatureVectorAndLabel());
        pipeList.add(new FeatureVector2FeatureVectorSequence());
	//Pipe crf_pipe = new CRFMalletFilePipe();
	// crf_pipe.getTargetAlphabet().lookupIndex(defaultLabel);
	// crf_pipe.setTargetProcessing(true);
        SerialPipes pipe  = new SerialPipes(pipeList);
	//add the pipe for creating feature vectors from text sentences
        System.out.println("piping");
	InstanceList trainingData = new InstanceList(pipe);
        //System.out.println("Training file: " + filename);
        System.out.println("through pipe");
	trainingData.addThruPipe(new LineGroupIterator(trainingFile,
                                                       Pattern.compile("^\\s*$"), true));	
        System.out.println("returning");
	return trainingData;
    }

    public void train() {
	try {
            String startName = crf.addOrderNStates(mTrainingData, null, null,
                                                   defaultLabel, null, null, true);
            
            //set an initial weight for all the features
            for (int i = 0; i < crf.numStates(); i++) {
                crf.getState(i).setInitialWeight (Transducer.IMPOSSIBLE_WEIGHT);
            }
            crf.getState(startName).setInitialWeight(0.0);
            
            //create a CRF trainer
            //CRFTrainerByLabelLikelihood crft = null;
            TransducerTrainer crft = null;
            //TransducerTrainer crft = null;
            if (classifier.equals("CRFTrainerByL1LabelLikelihood")) {
                CRFTrainerByL1LabelLikelihood c = new CRFTrainerByL1LabelLikelihood(crf);
                c.setGaussianPriorVariance(sigma*sigma);
                c.setL1RegularizationWeight(1.0);
                crft = c;
            } else  if (classifier.equals("CRFTrainerByLabelLikelihood")) {
                CRFTrainerByLabelLikelihood c = new CRFTrainerByLabelLikelihood(crf);
                c.setGaussianPriorVariance(sigma*sigma);
                crft = c;
            } else  if (classifier.equals("CRFTrainerByEntropyRegularization")) {
                CRFTrainerByEntropyRegularization c = new CRFTrainerByEntropyRegularization(crf);
                c.setGaussianPriorVariance(sigma*sigma);
                c.setEntropyWeight(0.5);
            } else {
                throw new IllegalStateException("Bad classifier: " + classifier);
            }
            
            InstanceList trainingData = mTrainingData;
            for (Instance i : trainingData) {
                // System.out.println("i: " + i.getData().getClass() + " " + 
                // i.getData());
            }
            
            boolean converged = false;
            for (int i = 1; i <= iterations; i++) {
                if (classifier.equals("CRFTrainerByEntropyRegularization")) {
                    //converged = crft.train(trainingData, unlabeledData, 1);
                } else { 
                    converged = crft.train(trainingData, 1);
                }
                if (converged) {
                    break;
                }
            }
        } catch (RuntimeException re) {
            re.printStackTrace();
            throw re;
        } catch (Error e) {
            e.printStackTrace();
            throw e;
        }
    }
    
    public void loadModel(String filename) 
	throws IOException, FileNotFoundException, ClassNotFoundException{
	File loadFile = new File(filename);
	ObjectInputStream s =
	    new ObjectInputStream(new FileInputStream(loadFile));
	crf = (CRF) s.readObject();
	s.close();
    }


    public void saveModel(String filename)throws IOException{
	File myfile = new File(filename);
	crf.write(myfile);
    }

    
    public FeatureVectorSequence stringArrayToFeatureVectorSequence(String[][] input){
	FeatureVector[] fva = new FeatureVector[input.length];

	Alphabet dAlphabet = crf.getInputAlphabet();

	for(int i=0; i<input.length; i++){
	    int featureIndices[] = new int[input[i].length];
	    
	    for(int j=0; j<input[i].length; j++){
		featureIndices[j] = dAlphabet.lookupIndex(input[i][j]);
	    }
	    fva[i] = new FeatureVector(dAlphabet, featureIndices);
	}

	FeatureVectorSequence fvs = new FeatureVectorSequence(fva);
	return fvs;
    }


    public FeatureVectorSequence arraysToFeatureVectorSequence(String[] features, double[] values) {

	FeatureVector[] fva = new FeatureVector[1];

	Alphabet dAlphabet = crf.getInputAlphabet();
        int featureIndices[] = new int[features.length];
        double featureValues[] = new double[features.length];

        for(int i = 0; i < features.length; i++){
            featureIndices[i] = dAlphabet.lookupIndex(features[i]);
            featureValues[i] = values[i];
        }
        fva[0] = new FeatureVector(dAlphabet, featureIndices, featureValues);

	FeatureVectorSequence fvs = new FeatureVectorSequence(fva);
	return fvs;
    }


    public LabelSequence stringArrayToLabelSequence(String[] labels){
	Alphabet dAlphabet = crf.getOutputAlphabet();
	
	int labelIndices[] = new int[labels.length];
	for(int i=0; i<labels.length; i++)
	    labelIndices[i] = dAlphabet.lookupIndex(labels[i]);

	LabelSequence fva = new LabelSequence((LabelAlphabet)dAlphabet, labelIndices);
	return fva;
    }
    
    public String[] sequenceToStringArray(Sequence fvs){
	String[] myString = new String[fvs.size()];
	
	for(int i=0; i<fvs.size(); i++){
	    myString[i] = (String)fvs.get(i);
	}	
	return myString;
    }
   
    /**
     * Apply a transducer to an input sequence to produce the k highest-scoring
     * output sequences.
     *
     * @param input the input sequence
     * @param k the number of answers to return
     * @return array of the k highest-scoring output sequences
     */
 
    public String[] predict(String[] keys, double[] values) {
        try {
            FeatureVectorSequence inputTS = arraysToFeatureVectorSequence(keys, values);
            Sequence[] answers = new Sequence[1];
            answers[0] = crf.transduce(inputTS);
            String[] answersStr = sequenceToStringArray(answers[0]);
            return answersStr;
        } catch (RuntimeException re) {
            re.printStackTrace();
            throw re;
        } catch (AssertionError ae) {
            ae.printStackTrace();
            throw ae;
        }
    }

    /**
     * Apply a transducer to an input sequence to produce the probability
     * of a particular 
     *
     * @param input the input sequence
     * @param k the number of answers to return
     * @return array of the k highest-scoring output sequences
     */
 
    public double log_probability(String[] output, String[] keys, double[] values) {
        try {
            FeatureVectorSequence inputTS = arraysToFeatureVectorSequence(keys, values);
            LabelSequence outputLS = stringArrayToLabelSequence(output);
            double logScore = new SumLatticeDefault(crf,inputTS,outputLS).getTotalWeight();
            double logZ = new SumLatticeDefault(crf,inputTS).getTotalWeight();
            double log_probability = logScore - logZ;
            
            return log_probability;
        } catch (Error t) {
            t.printStackTrace();
            throw t;
        }
    }

    public String[] featureNames() {
        SparseVector[] weights = crf.getParameters().weights;

        String[] out = new String[weights[0].getIndices().length];
        int idx = 0;
        for (int j : weights[0].getIndices()) {
            String name = (String) crf.getInputAlphabet().lookupObject(j);
            double value1 = weights[0].value(j);
            double value2 = weights[1].value(j);
            if (Math.abs(value1 + value2) > 0.000001) {
                System.out.println("Value not equal for " + name);
                System.out.println("Value 1: " + value1);
                System.out.println("Value 2: " + value2);
                throw new IllegalArgumentException();
            }
            out[idx] = name;
            idx++;
        }
        return out;
    }
    
    public double[] featureWeights() {
        SparseVector[] weights = crf.getParameters().weights;

        double[] out = new double[weights[1].getIndices().length];
        int idx = 0;
        for (int j : weights[1].getIndices()) {

            String name = (String) crf.getInputAlphabet().lookupObject(j);
            double value1 = weights[0].value(j);
            double value2 = weights[1].value(j);
            if (Math.abs(value1 + value2) > 0.000001) {
                System.out.println("Value not equal for " + name);
                System.out.println("Value 1: " + value1);
                System.out.println("Value 2: " + value2);
                throw new IllegalArgumentException();
            }
            out[idx] = weights[crf.getOutputAlphabet().lookupIndex("True")].value(j);
            idx++;
        }
        return out;
    }
}
