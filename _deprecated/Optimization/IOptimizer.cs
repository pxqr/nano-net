using System;

namespace Optimization
{
	public interface IOptimizer
	{
		void Minimize(Vector hypothesis, Vector[] inputs, double[] outputs);
	}
}

