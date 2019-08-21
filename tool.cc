#include <sanitizer.h>
#include <iostream>


#define SANITIZER_CALL(fn)  \
{  \
  SanitizerResult status = fn;  \
  if (status != SANITIZER_SUCCESS) {  \
    std::cout << "sanitizer error" << std::endl;  \
  }  \
}


void callback
(
  void* userdata,
  Sanitizer_CallbackDomain domain,
  Sanitizer_CallbackId cbid,
  const void* cbdata
) {
  if (domain == SANITIZER_CB_DOMAIN_RESOURCE && cbid == SANITIZER_CBID_RESOURCE_MODULE_LOADED) {
    auto* data = (Sanitizer_ResourceModuleData*)cbdata;
    SANITIZER_CALL(sanitizerAddPatchesFromFile("../../instrumentor.fatbin", 0));
    SANITIZER_CALL(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_MEMORY_ACCESS,
        data->module, "sanitizer_memory_access_callback"));
    SANITIZER_CALL(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BARRIER,
        data->module, "sanitizer_barrier_callback"));
    SANITIZER_CALL(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_SHFL,
        data->module, "sanitizer_block_exit_callback"));
    SANITIZER_CALL(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_ENTER,
        data->module, "sanitizer_block_enter_callback"));
    SANITIZER_CALL(sanitizerPatchInstructions(SANITIZER_INSTRUCTION_BLOCK_EXIT,
        data->module, "sanitizer_block_exit_callback"));
    SANITIZER_CALL(sanitizerPatchModule(data->module));
    std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
    std::cout << "Sanitizer invoked" << std::endl;
    std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
  }
}


int init() {
  Sanitizer_SubscriberHandle handle;

  sanitizerSubscribe(&handle, callback, NULL);
  sanitizerEnableAllDomains(1, handle);

  return 0;
}


int _ret_ = init();
