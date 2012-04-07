using System;
using System.Collections.Generic;

namespace Nanon.Data
{
	public interface IDataSet<InputT, OutputT>
	{
		IEnumerable<InputT>  Inputs  { get; }
		IEnumerable<OutputT> Outputs { get; }
		
		// alist
		IEnumerable<Tuple<InputT, OutputT> > Set { get; }
		int Size { get; }
		
		InputT FirstInput { get; }
		OutputT FirstOutput { get; }
	}
}

