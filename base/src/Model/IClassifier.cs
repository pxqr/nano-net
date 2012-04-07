using System;

namespace Nanon.Model.Classifier
{
	public interface IClassifier<InputT, OutputT>
	{
		OutputT Classify(InputT input);
	}
}