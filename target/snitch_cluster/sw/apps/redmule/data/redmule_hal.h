// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Yvan Tortorella <yvan.tortorella@unibo.it>
//

#ifndef __HAL_REDMULE_H__
#define __HAL_REDMULE_H__

/* LOW-LEVEL HAL */
#define REDMULE_ADDR_BASE REDMULE_BASE_ADD
#define REDMULE_ADDR_SPACE 0x00000100

#define HWPE_WRITE(value, offset) *(volatile int *)(REDMULE_ADDR_BASE + offset) = value
#define HWPE_READ(offset) *(volatile int *)(REDMULE_ADDR_BASE + offset)

static inline void redmule_x_add_set(unsigned int value) {
  HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_REG_X_PTR);
}

static inline void redmule_w_add_set(unsigned int value) {
  HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_REG_W_PTR);
}

static inline void redmule_z_add_set(unsigned int value) {
  HWPE_WRITE(value, REDMULE_REG_OFFS + REDMULE_REG_Z_PTR);
}

static inline void redmule_mcfg_set(uint32_t mcfg0, uint32_t mcfg1) {
  HWPE_WRITE(mcfg0, REDMULE_REG_OFFS + REDMULE_MCFG0_PTR);
  HWPE_WRITE(mcfg1, REDMULE_REG_OFFS + REDMULE_MCFG1_PTR);
}

static inline void redmule_arith_set(uint32_t arith) {
  HWPE_WRITE(arith, REDMULE_REG_OFFS + REDMULE_ARITH_PTR);
}

static inline void hwpe_trigger_job() { HWPE_WRITE(0, REDMULE_TRIGGER); }

static inline int hwpe_acquire_job() { return HWPE_READ(REDMULE_ACQUIRE); }

static inline unsigned int hwpe_get_status() { return HWPE_READ(REDMULE_STATUS); }

static inline void hwpe_soft_clear() {
  volatile int i;
  HWPE_WRITE(0, REDMULE_SOFT_CLEAR);
}

static inline void hwpe_evt_clear(int value) {
  HWPE_WRITE(value, HWPE_EVT_OFFS);
}

static inline void hwpe_cg_enable() { HWPE_WRITE(1, CK_GATE_OFFS); }

static inline void hwpe_cg_disable() { HWPE_WRITE(0, CK_GATE_OFFS); }

void redmule_cfg(unsigned int x, unsigned int w, unsigned int z, uint16_t m_size, uint16_t n_size,
                 uint16_t k_size, uint8_t gemm_op, uint8_t gemm_fmt) {

  uint32_t mcfg_reg0 = 0;
  uint32_t mcfg_reg1 = 0;
  uint32_t arith_reg = 0;

  mcfg_reg0 = (k_size << 16) | (m_size << 0);
  mcfg_reg1 = n_size << 0;

  arith_reg = (gemm_op << 10) | (gemm_fmt << 7);

  redmule_x_add_set((unsigned int)x);
  redmule_w_add_set((unsigned int)w);
  redmule_z_add_set((unsigned int)z);
  redmule_mcfg_set((unsigned int)mcfg_reg0, (unsigned int)mcfg_reg1);
  redmule_arith_set((unsigned int)arith_reg);
}

#define ERR 0x0011

int redmule16_compare_int(uint32_t *actual_z, uint32_t *golden_z, int len) {
  uint32_t actual_word = 0;
  uint16_t actual_MSHWord, actual_LSHWord;
  uint32_t golden_word = 0;
  uint16_t golden_MSHWord, golden_LSHWord;
  uint32_t actual = 0;
  uint32_t golden = 0;

  int errors = 0;
  int error;

  for (int i = 0; i < len; i++) {
    error = 0;
    actual_word = *(actual_z + i);
    golden_word = *(golden_z + i);

    // int error = ((actual_word ^ golden_word) & ~IGNORE_BITS_COMPARE) ? 1 : 0;
    uint16_t diff = 0;

    // Chechink Least Significant Half-Word
    actual_LSHWord = (uint16_t)(actual_word & 0x0000FFFF);
    golden_LSHWord = (uint16_t)(golden_word & 0x0000FFFF);

    diff = (actual_LSHWord > golden_LSHWord)   ? (actual_LSHWord - golden_LSHWord)
           : (actual_LSHWord < golden_LSHWord) ? (golden_LSHWord - actual_LSHWord)
                                               : 0;

    if (diff > ERR) {
      error = 1;
#ifdef VERBOSE
      printf("diff: 0x%08x\n", diff);
      printf("LSW: Error!\n");
#endif
    }

    // Checking Most Significant Half-Word
    actual_MSHWord = (uint16_t)((actual_word >> 16) & 0x0000FFFF);
    golden_MSHWord = (uint16_t)((golden_word >> 16) & 0x0000FFFF);

    diff = (actual_MSHWord > golden_MSHWord)   ? (actual_MSHWord - golden_MSHWord)
           : (actual_MSHWord < golden_MSHWord) ? (golden_MSHWord - actual_MSHWord)
                                               : 0;

    if (diff > ERR) {
      error = 1;
#ifdef VERBOSE
      printf("diff: 0x%08x\n", diff);
      printf("MSW: Error!\n");
#endif
    }

    errors += error;

#ifdef DEBUG
    printf("Golden: 0x%08x; Actual: 0x%08x,\n", golden_word, actual_word);
#endif

#ifdef VERBOSE
    if (error) {
      if (errors == 1) printf("  golden     <- actual     @ address    @ index\n");
      printf("0x%08x <- 0x%08x @ 0x%08x @ 0x%08x\n", golden_word, actual_word, (actual_z + i),
                 i * 4);
    }
#endif
  }
  return errors;
}

int redmule8_compare_int(uint32_t *actual_z, uint32_t *golden_z, int len) {
  uint32_t actual_word = 0;
  uint8_t actual_Byte0, actual_Byte1, actual_Byte2, actual_Byte3;
  uint32_t golden_word = 0;
  uint8_t golden_Byte0, golden_Byte1, golden_Byte2, golden_Byte3;
  uint32_t actual = 0;
  uint32_t golden = 0;

  int errors = 0;
  int error;

  for (int i = 0; i < len; i++) {
    error = 0;
    actual_word = *(actual_z + i);
    golden_word = *(golden_z + i);

    // int error = ((actual_word ^ golden_word) & ~IGNORE_BITS_COMPARE) ? 1 : 0;
    uint8_t diff = 0;

    // Cheching Byte0
    actual_Byte0 = (uint8_t)(actual_word & 0x000000FF);
    golden_Byte0 = (uint8_t)(golden_word & 0x000000FF);

    diff = (actual_Byte0 > golden_Byte0)   ? (actual_Byte0 - golden_Byte0)
           : (actual_Byte0 < golden_Byte0) ? (golden_Byte0 - actual_Byte0)
                                           : 0;

    if (diff > ERR) {
      error = 1;
#ifdef VERBOSE
      printf("diff: 0x%08x\n", diff);
      printf("Byte0: Error!\n");
#endif
    }

    // Cheching Byte1
    actual_Byte1 = (uint8_t)((actual_word >> 8) & 0x000000FF);
    golden_Byte1 = (uint8_t)((golden_word >> 8) & 0x000000FF);

    diff = (actual_Byte1 > golden_Byte1)   ? (actual_Byte1 - golden_Byte1)
           : (actual_Byte1 < golden_Byte1) ? (golden_Byte1 - actual_Byte1)
                                           : 0;

    if (diff > ERR) {
      error = 1;
#ifdef VERBOSE
      printf("diff: 0x%08x\n", diff);
      printf("Byte1: Error!\n");
#endif
    }

    // Cheching Byte2
    actual_Byte2 = (uint8_t)((actual_word >> 16) & 0x000000FF);
    golden_Byte2 = (uint8_t)((golden_word >> 16) & 0x000000FF);

    diff = (actual_Byte2 > golden_Byte2)   ? (actual_Byte2 - golden_Byte2)
           : (actual_Byte2 < golden_Byte2) ? (golden_Byte2 - actual_Byte2)
                                           : 0;

    if (diff > ERR) {
      error = 1;
#ifdef VERBOSE
      printf("diff: 0x%08x\n", diff);
      printf("Byte2: Error!\n");
#endif
    }

    // Cheching Byte3
    actual_Byte3 = (uint8_t)((actual_word >> 24) & 0x000000FF);
    golden_Byte3 = (uint8_t)((golden_word >> 24) & 0x000000FF);

    diff = (actual_Byte3 > golden_Byte3)   ? (actual_Byte3 - golden_Byte3)
           : (actual_Byte3 < golden_Byte3) ? (golden_Byte3 - actual_Byte3)
                                           : 0;

    if (diff > ERR) {
      error = 1;
#ifdef VERBOSE
      printf("diff: 0x%08x\n", diff);
      printf("Byte3: Error!\n");
#endif
    }

    errors += error;

#ifdef DEBUG
    printf("Golden: 0x%08x; Actual: 0x%08x,\n", golden_word, actual_word);
#endif

#ifdef VERBOSE
    if (error) {
      if (errors == 1) printf("  golden     <- actual     @ address    @ index\n");
      printf("  0x%08x <- 0x%08x @ 0x%08x @ 0x%08x\n", golden_word, actual_word, (actual_z + i),
                 i * 4);
    }
#endif
  }
  return errors;
}

#endif
