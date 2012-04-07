using System;
using System.Collections.Generic;

using Nanon.Model;

namespace Nanon.Learning.Optimization
{
	public interface IOptimizer<InputT, OutputT> 
	{
		void Optimize(IHypothesis<InputT, OutputT> hypothesis, IEnumerable<Tuple<InputT, OutputT> > exsamples);
	}
}

