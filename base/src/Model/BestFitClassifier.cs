using System;

using Nanon.Model.Regression;
using Nanon.Math.Linear;

namespace Nanon.Model.Classifier
{
	public class MaxFitClassifier<InputT> : IClassifier<InputT, int>
	{
		IRegression<InputT, Vector> regression;
			
		public MaxFitClassifier(IRegression<InputT, Vector> regressionArg)
		{
			regression = regressionArg;
		}
		
		public int Classify(InputT input)
		{
			return regression.Predict(input).IndexOfMax;
		}
	}
}

