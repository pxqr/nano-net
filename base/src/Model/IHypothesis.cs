using Nanon.Math.Linear;

namespace Nanon.Model
{
	//
	// hypothesis is the interface for trainable instance
	// which can become a trained regression or not (in worst case)
	//
	public interface IHypothesis<InputT, OutputT>
	{
		// returns value which could tell how close 
		// current hypothesis to ideal hypothesis
		double Cost(InputT input, OutputT output);
		
		// to recreate internal gradient structure
		Vector[] ZeroGradient { get; } 
		
		// find gradient of cost(loss) function
		// should get gradients for current "input, output"
		Vector[] Gradient(InputT input, OutputT output);
		
		// correct hypothesis by gradient
		void Correct(Vector[] gradient);
	}
}

