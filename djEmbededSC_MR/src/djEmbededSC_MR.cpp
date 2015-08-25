/*
 Sparse Coordinate Coding  version 1.0.2
 */
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <iomanip>
#include <string>
#include <omp.h>
#include "DictionaryGeneration.h"
#include "SampleNormalization.h"
#include "LR.h"
#include "SCC.h"
using namespace std;

int main(int argc, char* argv[]) {

	//***********************General Defination*******************************************//
	int layers = 3;
	int epochNumber = 3; // Experienced based
	int featureNumber = 400;
	int sampleElementNumber = 284;
	double lambda = 0.08;
	bool NonNegativeState = false;

	//***********************Input*******************************************//
	if (argc == 4) {
		string strSubID = argv[1]; //0-399
		int subID = atoi(strSubID.c_str());
		string strRoundIndex = argv[2]; //0-399
		int nRoundIndex = atoi(strRoundIndex.c_str());
		string strTaskName = argv[3]; //0-399

		//***********************Sparse Learning for each individual*******************************************//

		cout << "#################### Dealing with sub" << subID << endl;
		stringstream signalname;
		signalname
				<< "/ifs/loni/faculty/thompson/four_d/dzhu/data/HCP/TaskFMRI/Whole_b_signals/"<<strTaskName<<"/"
				<< subID << ".MOTOR.sig.txt";
		char SampleFileName[100];
		signalname >> SampleFileName;

		double **sample;
		int sampleNumber = dpl::getSampleNumber(SampleFileName);
		int iterationNumber = sampleNumber * epochNumber;

		std::cout << "Number of samples is " << sampleNumber << std::endl;
		std::cout << "Number of samples' element is " << sampleElementNumber
				<< std::endl;
		std::cout << "Number of features is " << featureNumber << std::endl;
		std::cout << "Number of Iterations is " << iterationNumber << std::endl;
		std::cout << "lambda is " << lambda << std::endl;
		std::cout << "subID is " << subID << std::endl;
		std::cout << "nRoundIndex is " << nRoundIndex << std::endl;

		std::cout << "Begin to read sample." << std::endl;
		sample = dpl::ReadSample(SampleFileName, sampleNumber,
				sampleElementNumber);
		std::cout << "Begin to normalize sample." << std::endl;
		dpl::SampleNormalization(sample, sampleNumber, sampleElementNumber);

		double **Wd;
		double **feature;
		stringstream Dname;
		stringstream Aname;
		stringstream Recordname;
		Wd = dpl::GenerateRandomPatchDictionary(featureNumber,
				sampleElementNumber, sampleNumber, sample);
		Dname
				<< "/ifs/loni/faculty/thompson/four_d/dzhu/data/HCP/TaskFMRI/multipleRuns/"<<strTaskName<<"/"
				<< subID << "/sub_" << subID << "_Round_" << nRoundIndex
				<< "_D.txt";

		char savedDictionaryName[100];
		Dname >> savedDictionaryName;
		//Initialize random dictionary
		Wd = dpl::GenerateRandomPatchDictionary(featureNumber,
				sampleElementNumber, sampleNumber, sample);
		dpl::DictionaryNormalization(featureNumber, sampleElementNumber, Wd);

		//Begin Sparse Learning
		feature = dpl::FeatureInitialization(featureNumber, sampleNumber);
		std::cout << "Begin to train " << std::endl;
		dpl::trainDecoder(Wd, feature, sample, lambda, layers, featureNumber,
				sampleNumber, sampleElementNumber, iterationNumber,
				NonNegativeState);
		std::cout << "Finish training " << std::endl;

		dpl::saveDictionary(featureNumber, sampleElementNumber, Wd,
				savedDictionaryName);

		dpl::clearFeature(sampleNumber, feature);
		dpl::clearDictionary(sampleElementNumber, Wd);

		dpl::clearSample(sampleNumber, sample);
		std::cout << "Hello World!" << std::endl;
		//    } //for all subjects
		return 0;
	} //if
	else
		cout
				<< "Need paramaters: subID(1-68) roundIndex strTaskName"
				<< endl;

}
