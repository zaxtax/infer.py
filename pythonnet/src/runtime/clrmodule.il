// ==========================================================================
// This software is subject to the provisions of the Zope Public License,
// Version 2.0 (ZPL).  A copy of the ZPL should accompany this distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY AND ALL EXPRESS OR IMPLIED
// WARRANTIES ARE DISCLAIMED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF TITLE, MERCHANTABILITY, AGAINST INFRINGEMENT, AND FITNESS
// FOR A PARTICULAR PURPOSE.
// ==========================================================================
//============================================================================
// This file is a hand-maintained stub - it implements clr.dll, which can be
// loaded by a standard CPython interpreter as an extension module. When it
// is loaded, it bootstraps the managed runtime integration layer and defers
// to it to do initialization and put the clr module into sys.modules, etc.
// The "USE_PYTHON_RUNTIME_*" defines control what extra evidence is used
// to help the CLR find the appropriate Python.Runtime assembly.
// If defined, the "pythonRuntimeVersionString" variable must be set to
// Python.Runtime's current version.
// If defined, the "PythonRuntimePublicKeyTokenData" data array must be
// set to Python.Runtime's public key token.
//#define USE_PYTHON_RUNTIME_PUBLIC_KEY_TOKEN
// If DEBUG_PRINT is defined, a few System.Console.WriteLine calls are made
// to indicate what's going on during the load...
//#define DEBUG_PRINT
//============================================================================
.assembly extern mscorlib
{
 .publickeytoken = (B7 7A 5C 56 19 34 E0 89 )
 .ver 2:0:0:0
}
.assembly clr
{
 .hash algorithm 0x00008004
 .ver 2:0:0:2
}
.module clr.dll
.imagebase 0x00400000
.subsystem 0x00000003
.file alignment 512
// This includes the platform-specific IL.  The include search path
// is set depending on whether we're compiling 32 or 64 bit.
// This MUST come before any other .data directives!
// Why, oh why, can't ilasm support command line #defines? :(
// Contributed by VIKAS DHIMAN - Handled by /home/barton/Projects/PyDotNet/pythonnet/makefile
// gcc -C -P -x c++ -I $(ARCH) clrmodule.pp.il -o clrmodule.il
// to copy the correct architecture to the clrModule.
// Nice formating, as well - Thanks, Vikas!
// ==========================================================================
// This software is subject to the provisions of the Zope Public License,
// Version 2.0 (ZPL).  A copy of the ZPL should accompany this distribution.
// THIS SOFTWARE IS PROVIDED "AS IS" AND ANY AND ALL EXPRESS OR IMPLIED
// WARRANTIES ARE DISCLAIMED, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF TITLE, MERCHANTABILITY, AGAINST INFRINGEMENT, AND FITNESS
// FOR A PARTICULAR PURPOSE.
// ==========================================================================
.vtfixup [1] int32 fromunmanaged at VT_01
.data VT_01 = int32(0)
.class public auto ansi beforefieldinit clrModule extends [mscorlib]System.Object
{
 .method public hidebysig specialname rtspecialname instance void
  .ctor() cil managed
 {
  .maxstack 1
  ldarg.0
  call instance void [mscorlib]System.Object::.ctor()
  ret
 }
 .method public hidebysig static void modopt([mscorlib]System.Runtime.CompilerServices.CallConvCdecl)
  initclr() cil managed
 {
  .vtentry 1:1
  .export [1] as initclr
  .maxstack 6
  .locals init (
   class [mscorlib]System.Reflection.Assembly pythonRuntime,
   class [mscorlib]System.Reflection.Assembly executingAssembly,
   class [mscorlib]System.Reflection.AssemblyName pythonRuntimeName,
   class [mscorlib]System.Type pythonEngineType,
   int8[] publicKeyToken,
   string assemblyDirectory,
   string pythonRuntimeVersionString,
   string pythonRuntimeDllPath)
  // pythonRuntime = null;
  ldnull
  stloc pythonRuntime
  .try
  {
   // Attempt to find and load Python.Runtime using standard assembly binding rules.
   // This roughly translates into looking in order:
   // - GAC
   // - ApplicationBase
   // - A PrivateBinPath under ApplicationBase
   // With an unsigned assembly, the GAC is skipped.
   // System.Reflection.AssemblyName pythonRuntimeName = new System.Reflection.AssemblyName();
   newobj instance void [mscorlib]System.Reflection.AssemblyName::.ctor()
   stloc pythonRuntimeName
   // pythonRuntimeName.Name = "Python.Runtime";
   ldloc pythonRuntimeName
   ldstr "Python.Runtime"
   callvirt instance void [mscorlib]System.Reflection.AssemblyName::set_Name(string)
   // pythonRuntimeVersionString = "...";
   ldstr "2.0.0.2"
   stloc pythonRuntimeVersionString
   // pythonRuntimeName.Version = new Version(pythonRuntimeVersionString);
   ldloc pythonRuntimeName
   ldloc pythonRuntimeVersionString
   newobj instance void [mscorlib]System.Version::.ctor(string)
   callvirt instance void [mscorlib]System.Reflection.AssemblyName::set_Version(class [mscorlib]System.Version)
   // pythonRuntimeName.CultureInfo = System.Globalization.CultureInfo.InvariantCulture;
   ldloc pythonRuntimeName
   call class [mscorlib]System.Globalization.CultureInfo [mscorlib]System.Globalization.CultureInfo::get_InvariantCulture()
   callvirt instance void [mscorlib]System.Reflection.AssemblyName::set_CultureInfo(class [mscorlib]System.Globalization.CultureInfo)
   // return System.Reflection.Assembly.Load(pythonRuntimeName);
   ldloc pythonRuntimeName
   call class [mscorlib]System.Reflection.Assembly [mscorlib]System.Reflection.Assembly::Load(class [mscorlib]System.Reflection.AssemblyName)
   stloc pythonRuntime
   leave.s LOADED_PYTHON_RUNTIME
  }
  catch [mscorlib]System.Object
  {
   leave.s EXIT_CLR_LOAD
  }
  EXIT_CLR_LOAD: nop
  .try
  {
   // If the above fails for any reason, we fallback to attempting to load "Python.Runtime.dll"
   // from the directory this assembly is running in. "This assembly" is probably "clr.pyd",
   // sitting somewhere in PYTHONPATH.  This is using Assembly.LoadFrom, and inherits all the
   // caveats of that call.  See MSDN docs for details.
   // Suzanne Cook's blog is also an excellent source of info on this:
   // http://blogs.msdn.com/suzcook/
   // http://blogs.msdn.com/suzcook/archive/2003/05/29/57143.aspx
   // http://blogs.msdn.com/suzcook/archive/2003/06/13/57180.aspx
   // executingAssembly = System.Reflection.Assembly.GetExecutingAssembly();
   call class [mscorlib]System.Reflection.Assembly [mscorlib]System.Reflection.Assembly::GetExecutingAssembly()
   stloc executingAssembly
   // assemblyDirectory = System.IO.Path.GetDirectoryName(executingAssembly.Location);
   ldloc executingAssembly
   callvirt instance string [mscorlib]System.Reflection.Assembly::get_Location()
   call string [mscorlib]System.IO.Path::GetDirectoryName(string)
   stloc assemblyDirectory
   // pythonRuntimeDllPath = System.IO.Path.Combine(assemblyDirectory, "Python.Runtime.dll");
   ldloc assemblyDirectory
   ldstr "Python.Runtime.dll"
   call string [mscorlib]System.IO.Path::Combine(string, string)
   stloc pythonRuntimeDllPath
   // pythonRuntime = System.Reflection.Assembly.LoadFrom(pythonRuntimeDllPath);
   ldloc pythonRuntimeDllPath
   call class [mscorlib]System.Reflection.Assembly [mscorlib]System.Reflection.Assembly::LoadFrom(string)
   stloc pythonRuntime
   leave.s LOADED_PYTHON_RUNTIME
  }
  catch [mscorlib]System.Object
  {
   leave.s EXIT_PYTHONPATH_LOAD
  }
  EXIT_PYTHONPATH_LOAD: nop
  // If we get here, we haven't loaded Python.Runtime, so bail.
  ret
  // Once here, we've successfully loaded SOME version of Python.Runtime
  // So now we get the PythonEngine and execute the InitExt method on it.
  LOADED_PYTHON_RUNTIME: nop
  .try
  {
   // pythonEngineType = pythonRuntime.GetType("Python.Runtime.PythonEngine");
   ldloc pythonRuntime
   ldstr "Python.Runtime.PythonEngine"
   callvirt instance class [mscorlib]System.Type [mscorlib]System.Reflection.Assembly::GetType(string)
   stloc pythonEngineType
   // pythonEngineType.InvokeMember("InitExt", System.Reflection.BindingFlags.InvokeMethod, null, null, null);
   ldloc pythonEngineType
   ldstr "InitExt"
   ldc.i4 0x100
   ldnull
   ldnull
   ldnull
   callvirt instance object [mscorlib]System.Type::InvokeMember( string,
                    valuetype [mscorlib]System.Reflection.BindingFlags,
                    class [mscorlib]System.Reflection.Binder,
                    object,
                    object[])
   pop
   leave.s EXIT_TRY_INVOKE
  }
  catch [mscorlib]System.Object
  {
   leave.s EXIT_TRY_INVOKE
  }
  EXIT_TRY_INVOKE: nop
  ret
 }
}
