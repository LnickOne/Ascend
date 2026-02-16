add_library(ascendc_runtime_obj OBJECT IMPORTED)
set_target_properties(ascendc_runtime_obj PROPERTIES
    IMPORTED_OBJECTS "/home/lnick/GitHub/Ascend/AscendC/lesson_01/0_introduction/0_helloworld/build/ascendc_runtime.cpp.o;/home/lnick/GitHub/Ascend/AscendC/lesson_01/0_introduction/0_helloworld/build/aicpu_rt.cpp.o;/home/lnick/GitHub/Ascend/AscendC/lesson_01/0_introduction/0_helloworld/build/ascendc_elf_tool.c.o"
)
