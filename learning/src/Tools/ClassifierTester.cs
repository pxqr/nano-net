using System;

using Nanon.Data;
using Nanon.Model.Classifier;

namespace Nanon.Learning.Tools
{
	public class ClassifierTester<InputT, OutputT, OutputC> : ITester<InputT, OutputT>
	{
		IClassifier<InputT, OutputC> classifier;
		Func<OutputT, OutputC>       conv;
		
		public ClassifierTester(IClassifier<InputT, OutputC> classifierA, Func<OutputT, OutputC> convA)
		{
			classifier = classifierA;
			conv = convA;
		}
		
		public double Test(IDataSet<InputT, OutputT> dataSet)
		{
			int setSize = dataSet.Size;
			
			if (setSize == 0)
				return 0.0d;
			
			int correctCount = 0;
			
			foreach(var x in dataSet.Set)
			{
				var prediction = classifier.Classify(x.Item1);
				var answer = conv(x.Item2); 
			    if (prediction.Equals(answer))
					++correctCount;
			}
				
			return (double)correctCount / (double)setSize;
		}
	}
}

