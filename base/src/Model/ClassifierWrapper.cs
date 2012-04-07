using System;

namespace Nanon.Model.Classifier
{
	public class ClassifierWrapper<InputT, OutputT> : IClassifier<InputT, OutputT>
	{
		Func<InputT, OutputT> classifier;
		
		public ClassifierWrapper (Func<InputT, OutputT> clf)
		{
			classifier = clf;
		}

		#region IClassifier[InputT,OutputT] implementation
		
		public OutputT Classify (InputT input)
		{
			return classifier(input);
		}

		#endregion
	}
}

