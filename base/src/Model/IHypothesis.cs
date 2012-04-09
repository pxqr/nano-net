using Nanon.Math.Linear;

using Nanon.Model.Regression;

namespace Nanon.Model
{
	//
	// hypothesis is the interface for trainable instance
	// which can become a trained regression or not (in worst case)
	//
	public interface IHypothesis<InputT, OutputT> : IRegression<InputT, OutputT>
	{		
		// find gradient of cost(loss) function
		// should get gradients for current "input, output"
		void Gradient(InputT input, OutputT output);
		
		// correct hypothesis by gradient
		void Correct(double coeff);
	}
}

