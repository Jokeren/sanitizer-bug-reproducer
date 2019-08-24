#include <sanitizer.h>
#include <iostream>


#define SANITIZER_CALL(fn)  \
{  \
  SanitizerResult status = fn;  \
  if (status != SANITIZER_SUCCESS) {  \
    std::cout << "sanitizer error" << std::endl;  \
  }  \
}


const size_t DATA_SIZE = 7;

int *lock_dev;
int lock_host[DATA_SIZE];

void callback
(
  void* userdata,
  Sanitizer_CallbackDomain domain,
  Sanitizer_CallbackId cbid,
  const void* cbdata
) {
  if (domain == SANITIZER_CB_DOMAIN_RESOURCE && cbid == SANITIZER_CBID_RESOURCE_MODULE_LOADED) {
    auto* data = (Sanitizer_ResourceModuleData*)cbdata;
    if (INSTRUMENTOR == 0) {
      SANITIZER_CALL(sanitizerAddPatchesFromFile("../../empty.fatbin", 0));
    } else if (INSTRUMENTOR == 1) {
      SANITIZER_CALL(sanitizerAddPatchesFromFile("../../cas.fatbin", 0));
    }
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
    std::cout << "Sanitizer module resource load" << std::endl;
  } else if (domain == SANITIZER_CB_DOMAIN_LAUNCH && cbid == SANITIZER_CBID_LAUNCH_BEGIN) {
    auto *data = (Sanitizer_LaunchData *)cbdata;
    if (INSTRUMENTOR == 1) {
      SANITIZER_CALL(sanitizerAlloc((void **)&lock_dev, DATA_SIZE * sizeof(int)));
      SANITIZER_CALL(sanitizerMemset(lock_dev, 0, DATA_SIZE * sizeof(int), data->stream));
      SANITIZER_CALL(sanitizerSetCallbackData(data->stream, lock_dev));
    }
    std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
    std::cout << "Sanitizer kernel launch" << std::endl;
  } else if (domain == SANITIZER_CB_DOMAIN_SYNCHRONIZE && cbid == SANITIZER_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED) {
    auto *data = (Sanitizer_SynchronizeData *)cbdata;
    if (INSTRUMENTOR == 1) {
      SANITIZER_CALL(sanitizerMemcpyDeviceToHost(lock_host, lock_dev, DATA_SIZE * sizeof(int), data->stream));
      for (size_t i = 0; i < DATA_SIZE; ++i) {
        printf("index %d : value %d\n", i, lock_host[i]);
      }
    }
    std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" << std::endl;
    std::cout << "Sanitizer synchronize" << std::endl;
  }
}


int init() {
  Sanitizer_SubscriberHandle handle;

  sanitizerSubscribe(&handle, callback, NULL);
  sanitizerEnableAllDomains(1, handle);

  return 0;
}


int _ret_ = init();
