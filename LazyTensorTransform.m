% LazyTensorTransform - CLASS Provide transparent lazy function evaluation on a tensor
%
% Usage: ltt = LazyTensorTransform(tfInput, fhTransformation)
%        ltt = LazyTensorTransform(...<, vnDim1Window, vnDim2Window, ...>)
%        ltt = LazyTensorTransform(...<, 'UseCache', bUseCache>, < 'UseMappedTensor', bUseMappedTensor>)
%
% 'tfInput' is an arbitrary tensor. 'fhTransformation' is a function that
% operates on data from 'tfInput'. 'ltt' will appear as a tensor of the
% same size as 'tfInput'; when 'ltt' is referenced, then 'fhTransformation'
% will be evaluated on 'tfInput' in a lazy manner.
%
% 'fhTransformation' can have the signature @(fData) or @(fData, cSubs).
% 'fData' will be the portion of the input tensor to operate on. 'cSubs',
% if required, will be a cell array of the input subscripts that correspond
% to the current evaluation of 'fhTransformation'.
%
% By default, 'fhTransformation' operates element-wise on 'tfInput'. If
% required, 'fhTransformation' can instead operate on a window of data from
% 'tfInput', relative to the subscripts of 'tfInput' being evaluated. In
% this case, the window indices relative to a given subscript must be
% provided when the LazyTensorTransform object is constructed:
%
% >> tfInput = rand(10, 10);
% >> ltt = LazyTensorTransform(tfInput, fhFun, [-1 0 1], [-1 0 1]);
% >> ltt(2, 2);
%
% This will evaluate 'fhFun' on a window of tfInput([1 2 3], [1 2 3]).
%
% LazyTensorTransform optionally provides a cache of evaluated results.
% This can speed up access for computationally intensive transformations.
% Specify this when creating the LazyTensorTransform object:
%
% >> ltt = LazyTensorTransform(tfInput, fhFun, 'UseCache', true);
%
% By default, no caching is performed. If a cache is being used,
% LazyTensorTransform can optionally use a MappedTensor object as the
% cache. Specify this when creating the LazyTensorTransform object:
%
% ltt = LazyTensorTransform(tfInput, fhFun, 'UseCache', true, 'UseMappedTensor', true);

% Author: Dylan Muir <dylan.muir@unibas.ch>
% Created: 23rd February, 2016

classdef LazyTensorTransform < matlab.mixin.Copyable
   
   properties (SetAccess = private)
      
   end
   
   properties (GetAccess = private, SetAccess = private)
      cvnIndexWindows;
      fhTransform;
      tfTensor;
      tfWindow;
      fhRepSum;
      vnTensorSize;
      cvnTensorSize;
      bUseCache;
      tfCache;
      tbCache;
      vbUnitaryWindows;
   end
   
   methods
      function t = LazyTensorTransform(tfTensor, fhTransform, varargin)
         % - Create object
         t = t@matlab.mixin.Copyable;
         
         % - Default flags
         t.bUseCache = false;
         bMappedTensor = false;
         
         % - Parse string arguments
         vbStringArg = cellfun(@isstr, varargin);
         
         for (nArg = find(vbStringArg))
            switch lower(varargin{nArg})
               case {'cache', 'usecache'}
                  if ((nArg == numel(varargin)) || ~islogical(varargin{nArg+1}) || ~isscalar(varargin{nArg+1}))
                     error('LazyTensorTransform:Usage', ...
                        '*** LazyTensorTransform: A logical scalar value is required for the %s argument.', varargin{nArg});
                  end
                  t.bUseCache = varargin{nArg+1};
                  vbStringArg(nArg+1) = true;
                  
               case {'mappedtensor', 'usemappedtensor'}
                  if ((nArg == numel(varargin)) || ~islogical(varargin{nArg+1}) || ~isscalar(varargin{nArg+1}))
                     error('LazyTensorTransform:Usage', ...
                        '*** LazyTensorTransform: A logical scalar value is required for the %s argument.', varargin{nArg});
                  end
                  bMappedTensor = varargin{nArg+1};
                  vbStringArg(nArg+1) = true;
                  
               otherwise
                  error('LazyTensorTransform:Usage', ...
                        '*** LazyTensorTransform: Unknown string argument [%s].', varargin{nArg});
            end
         end
         
         % - Remove string arguments
         varargin = varargin(~vbStringArg);
         
         % - Record evaulation window arguments
         cvnIndexWindows = varargin; %#ok<*PROP>
         nNumDims = ndims(tfTensor);
         
         % - Check number of evaluation windows
         if (numel(cvnIndexWindows) > nNumDims)
            error('LazyTensorTransform:Arguments', ...
               '*** LazyTensorTransform: [%d] evaluation windows provided; expected [%d].', ...
               numel(cvnIndexWindows), nNumDims);
         end
         
         % - Catch empty index window
         if (isempty(cvnIndexWindows))
            cvnIndexWindows{nNumDims} = 0;
         end
         
         % - Check evaulation windows
         t.vbUnitaryWindows = false(size(cvnIndexWindows));
         for (nDim = 1:nNumDims)
            % - If not provided, assume identity window
            if (isempty(cvnIndexWindows{nDim}))
               cvnIndexWindows{nDim} = 0;
            end
            
            % - Check for reasonable reference windows
            if (~isnumeric(cvnIndexWindows{nDim}) || ...
                  ~isreal(cvnIndexWindows{nDim}) || ...
                  any(isnan(cvnIndexWindows{nDim})) || ...
                  any(isinf(cvnIndexWindows{nDim})) || ...
                  any(mod(cvnIndexWindows{nDim}, 1)))
               error('LazyTensorTransform:Arguments', ...
                  '*** LazyTensorTransform: Evaluation windows must be integer values.');
            end
            
            % - Check for unitary windows
            if (isequal(cvnIndexWindows{nDim}, 0))
               t.vbUnitaryWindows(nDim) = true;
            end
         end
         
         % - Check input function
         if (nargin(fhTransform) < 1)
            error('LazyTensorTransform:Arguments', ...
               '*** LazyTensorTransform: Evaulation function must have the signature ''@(tfData)'' or ''@(tfData, cSubs)''.');
         end
         
         % - Permit transformation functions that accept only the data input
         if (nargin(fhTransform) < 2)
            fhTransform = @(tfData, cSubs)fhTransform(tfData);
         end
         
         % - Create object
         t.cvnIndexWindows = cvnIndexWindows;
         t.tfTensor = tfTensor;
         t.fhTransform = fhTransform;
         t.fhRepSum = LTT_GetMexFunctionHandles();
         t.vnTensorSize = size(tfTensor);
         t.cvnTensorSize = num2cell(t.vnTensorSize);
         
         % - Set up cache
         if (t.bUseCache)
            t.bUseCache = false;
            S.type = '()';
            S.subs = {1};
            strResultClass = class(subsref(t, S));
            t.bUseCache = true;
            
            if (bMappedTensor)
               t.tfCache = MappedTensor(t.vnTensorSize, 'Class', strResultClass);
            else
               t.tfCache = eval(sprintf('%s(zeros(t.vnTensorSize))', strResultClass));
            end
            
            t.tbCache = false(t.vnTensorSize);
         end
      end
      
      %% Overloaded size, numel, etc
      function varargout = size(lttObject, varargin)
         [varargout{1:nargout}] = size(lttObject.tfTensor, varargin{:});
      end
      
      function varargout = numel(lttObject)
         [varargout{1:nargout}] = numel(lttObject.tfTensor);
      end
      
      function varargout = class(lttObject)
         [varargout{1:nargout}] = class(lttObject.tfTensor);
      end
      
      function varargout = isnumeric(lttObject)
         [varargout{1:nargout}] = isnumeric(lttObject.tfTensor);
      end
      
      %% Overloaded subsref, subsasgn
      function tfResult = subsref(lttObject, S)
         switch S.type
            case '()'
               tfResult = LTT_Evaluate(lttObject, S);
               
            case '.'
               error('LazyTensorTransform:nonStrucReference', 'Attempt to reference field of non-structure array.');
               
            case '{}'
               error('LazyTensorTransform:cellRefFromNonCell', 'Cell contents reference from a non-cell array object.');
               
            otherwise
               error('LazyTensorTransform:UnsupportedRef', ...
                  '*** LazyTensorTransform: [%s] reference type is not supported.', S.type);
         end
      end
      
      function lttObject = subsasgn(lttObject, S, B)
         % - Call subsasgn on wrapped tensor
         lttObject.tfTensor = subsasgn(lttObject.tfTensor, S, B);
         
         % - Invalidate cache
         lttObject.tbCache(:) = false;
      end
   end
   
end

function tfResult = LTT_Evaluate(lttObject, S)
   vnTensorSize = lttObject.vnTensorSize;
   cvnTensorSize = lttObject.cvnTensorSize;
   vbUnitaryWindows = lttObject.vbUnitaryWindows;
   
   if (numel(S.subs) > 1)
      % - Validate subscripts
      S.subs = LTT_ValidateSubs(lttObject, S.subs);
      
      % - Get linear indices
      [tfResult, vnDimRefSizes] = LTT_GetLinearIndicesForRefs(S.subs, vnTensorSize, lttObject.fhRepSum);
      tfResult = reshape(tfResult, vnDimRefSizes);
      
   else
      if (max(cellfun(@max, S.subs)) > numel(lttObject.tfTensor))
         error('badsubscript');
      end
      
      if (min(cellfun(@min, S.subs)) < 1)
         error('badsubscript');
      end
      
      tfResult = S.subs{1};
      % vnDimRefSizes = size(S.subs{1});
   end
   
   % - Check cache for pre-evaluated values
   if (lttObject.bUseCache)
      vbInCache = lttObject.tbCache(tfResult);
      
      % - Loop over non-cached elements
      for (nRef = find(~vbInCache)')
         % - Collect evaluation window's worth of data and call evaluation function
         [cvnSubs{1:numel(vnTensorSize)}] = ind2sub(vnTensorSize, tfResult(nRef));
         lttObject = LTT_FillEvaulationWindow(lttObject, cvnSubs, cvnTensorSize, vbUnitaryWindows);
         lttObject.tfCache(tfResult(nRef)) = lttObject.fhTransform(lttObject.tfWindow, S.subs);
      end
      
      % - Retrieve from cache
      lttObject.tbCache(tfResult(~vbInCache)) = true;
      tfResult = lttObject.tfCache(tfResult);
      
   else
      % - Loop over referenced elements
      for (nRef = 1:numel(tfResult))
         % - Collect evaluation window's worth of data and call evaluation function
         [cvnSubs{1:numel(vnTensorSize)}] = ind2sub(vnTensorSize, tfResult(nRef));
         lttObject = LTT_FillEvaulationWindow(lttObject, cvnSubs, cvnTensorSize, vbUnitaryWindows);
         tfResult(nRef) = lttObject.fhTransform(lttObject.tfWindow, S.subs);
      end
   end
end

function lttObject = LTT_FillEvaulationWindow(lttObject, cSubs, cvnTensorSize, vbUnitaryWindows)
   % - Construct unitary window indices
   cvnWindows(vbUnitaryWindows) = cSubs(vbUnitaryWindows);
   cvbValidRefs(vbUnitaryWindows) = {true};
   cvbNanRefs(vbUnitaryWindows) = {false};
   
   % - Construct non-unitary window indices
   cvnWindows(~vbUnitaryWindows) = lttObject.cvnIndexWindows(~vbUnitaryWindows);
   for (nWindowDim = find(~vbUnitaryWindows))
      cvnWindows{nWindowDim} = cvnWindows{nWindowDim} + cSubs{nWindowDim};
      cvbValidRefs{nWindowDim} = (cvnWindows{nWindowDim} >= 1) & (cvnWindows{nWindowDim} <= cvnTensorSize{nWindowDim});
      cvbNanRefs{nWindowDim} = ~cvbValidRefs{nWindowDim};
      cvnWindows{nWindowDim} = cvnWindows{nWindowDim}(cvbValidRefs{nWindowDim});
   end
   
   % - Allocate window
   if isempty(lttObject.tfWindow)
      vnWindowSize = cellfun(@numel, cvnWindows);
      lttObject.tfWindow = nan(vnWindowSize);
   end
   
   % - Fill window
   lttObject.tfWindow(cvbValidRefs{:}) = lttObject.tfTensor(cvnWindows{:});
   lttObject.tfWindow(cvbNanRefs{:}) = nan;
end


function [cSubs, vnTensorSize] = LTT_ValidateSubs(lttObject, cSubs)
   % - Check valid subscripts
   cellfun(@isvalidsubscript, cSubs);
   
   % - Check ranges
   vnMinInd = cellfun(@min, cSubs);
   vnMaxInd = cellfun(@max, cSubs);
   vnTensorSize = size(lttObject.tfTensor);
   
   if (any(arrayfun(@(v)lt(v,1), vnMinInd)))
      error('LazyTensorTransform:badsubscript', 'Subscript indices must either be real positive integers or logicals.');
   end
   
   if (any(arrayfun(@gt, vnMaxInd, vnTensorSize)))
      error('LazyTensorTransform:badsubscript', 'Index exceeds matrix dimensions.');
   end
   
   % - Convert colons
   vnColonRefs = find(cellfun(@iscolon, cSubs));
   for (nDim = vnColonRefs)
      cSubs{nDim} = 1:vnTensorSize(nDim);
   end
end

% GetLinearIndicesForRefs - FUNCTION Convert a set of multi-dimensional indices directly into linear indices
function [vnLinearIndices, vnDimRefSizes] = LTT_GetLinearIndicesForRefs(cRefs, vnLims, hRepSumFunc)

   % - Find colon references
   vbIsColon = cellfun(@iscolon, cRefs);
   
   if (all(vbIsColon))
      vnLinearIndices = 1:prod(vnLims);
      vnDimRefSizes = vnLims;
      return;
   end
   
   nFirstNonColon = find(~vbIsColon, 1, 'first');
   vbTrailingRefs = true(size(vbIsColon));
   vbTrailingRefs(1:nFirstNonColon-1) = false;
   vnDimRefSizes = cellfun(@numel, cRefs);
   vnDimRefSizes(vbIsColon) = vnLims(vbIsColon);
   
   % - Calculate dimension offsets
   vnDimOffsets = [1 cumprod(vnLims)];
   vnDimOffsets = vnDimOffsets(1:end-1);

   % - Remove trailing "1"s
   vbOnes = cellfun(@(c)isequal(c, 1), cRefs);
   nLastNonOne = find(~vbOnes, 1, 'last');
   vbTrailingRefs((nLastNonOne+1):end) = false;

   % - Check reference limits
   if (any(cellfun(@(r,l)any(r>l), cRefs(~vbIsColon), num2cell(vnLims(~vbIsColon)))))
      error('TIFFStack:badsubscript', 'Index exceeds matrix dimensions.');
   end
   
   % - Work out how many linear indices there will be in total
   nNumIndices = prod(vnDimRefSizes);
   vnLinearIndices = zeros(nNumIndices, 1);
   
   % - Build a referencing window encompassing the leading colon refs (or first ref)
   if (nFirstNonColon > 1)
      vnLinearIndices(1:prod(vnLims(1:(nFirstNonColon-1)))) = 1:prod(vnLims(1:(nFirstNonColon-1)));
   else
      vnLinearIndices(1:vnDimRefSizes(1)) = cRefs{1};
      vbTrailingRefs(1) = false;
   end
   
   % - Replicate windows to make up linear indices
   for (nDimension = find(vbTrailingRefs & ~vbOnes))
      % - How long is the current window?
      nCurrWindowLength = prod(vnDimRefSizes(1:(nDimension-1)));
      nThisWindowLength = nCurrWindowLength * vnDimRefSizes(nDimension);
      
      % - Is this dimension a colon reference?
      if (vbIsColon(nDimension))
         vnLinearIndices(1:nThisWindowLength) = hRepSumFunc(vnLinearIndices(1:nCurrWindowLength), ((1:vnLims(nDimension))-1) * vnDimOffsets(nDimension));

      else
         vnLinearIndices(1:nThisWindowLength) = hRepSumFunc(vnLinearIndices(1:nCurrWindowLength), (cRefs{nDimension}-1) * vnDimOffsets(nDimension));
      end
   end
end

% iscolon - FUNCTION Test whether a reference is equal to ':'
function bIsColon = iscolon(ref)
   bIsColon = ischar(ref) && isequal(ref, ':');
end

% isvalidsubscript - FUNCTION Test whether a vector contains valid entries
% for subscript referencing
function isvalidsubscript(oRefs)
   try
      % - Test for colon
      if (iscolon(oRefs))
         return;
      end
      
      if (islogical(oRefs))
         % - Test for logical indexing
         validateattributes(oRefs, {'logical'}, {'binary'});
         
      else
         % - Test for normal indexing
         validateattributes(oRefs, {'single', 'double'}, {'integer', 'real', 'positive'});
      end
      
   catch
      error('MappedTensor:badsubscript', ...
            '*** MappedTensor: Subscript indices must either be real positive integers or logicals.');
   end
end

% mapped_tensor_repsum_nomex - FUNCTION Slow version of replicate and sum
function [vfDest] = mapped_tensor_repsum_nomex(vfSourceA, vfSourceB)
   [mfA, mfB] = meshgrid(vfSourceB, vfSourceA);
   vfDest = mfA(:) + mfB(:);
end

function [hRepSumFunc] = LTT_GetMexFunctionHandles
   % - Does the compiled MEX function exist?
   if (exist('mapped_tensor_repsum') ~= 3) %#ok<EXIST>
      try %#ok<TRYNC>
         % - Move to the MappedTensor private directory
         strMTDir = fileparts(which('LazyTensorTransform'));
         strCWD = cd(fullfile(strMTDir, 'private'));
         
         % - Try to compile the MEX functions
         disp('--- LazyTensorTransform: Compiling MEX functions.');
         mex('mapped_tensor_repsum.c', '-largeArrayDims', '-O');
         
         % - Move back to previous working directory
         cd(strCWD);
      end
   end
   
   % - Did we succeed?
   if (exist('mapped_tensor_repsum') == 3) %#ok<EXIST>
      hRepSumFunc = @mapped_tensor_repsum;
      
   else
      % - Just use the slow matlab version
      warning('LazyTensorTransform:MEXCompilation', ...
         '--- LazyTensorTransform: Could not compile MEX functions.  Using slow matlab versions.');
      
      hRepSumFunc = @mapped_tensor_repsum_nomex;
   end
end

